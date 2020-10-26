from distributed_worker import DistributedManager, DistributedWorker
from tensorboardX import SummaryWriter
from typing import Tuple, Any
import torch
import copy
import pprint
import numpy as np
import time

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def disable_grad(model):
    for param in model.parameters():
        param.requires_grad = False

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, device="cpu", **kwargs):
        super().__init__()
        self.device = torch.device(device)
        disable_grad(model)

        self.optimizer = kwargs.get('optimizer', None)
        self.model = model.to(self.device).eval()  # No gradients needed
        self.model_shapes = []
        self.NPARAMS = 0
        for param in self.model.parameters():
            self.model_shapes.append(param.data.shape)
            self.NPARAMS += np.product(param.data.shape)

        self.eval()

    # Adds to params
    def update_from_delta(self, delta):
        idx = 0
        i = 0
        for param in self.model.parameters():
            size = np.product(self.model_shapes[i])
            block = delta[idx:idx+size]
            block = block.view(self.model_shapes[i])
            i += 1
            idx += size
            param.data += block

    # Overwrites params
    def set_from_delta(self, delta):
        idx = 0
        i = 0
        for param in self.model.parameters():
            size = np.product(self.model_shapes[i])
            block = delta[idx:idx+size]
            block = block.view(self.model_shapes[i])
            i += 1
            idx += size
            param.data.copy_(block)

    def sign(self, a):
        return (a > 0) - (a < 0)

    def update_from_epoch(self, update, log=False, optimizer=None):
        seed_dict = update['deltas']

        delta = torch.zeros(
            (self.NPARAMS,), requires_grad=False, device=self.device)

        sigma = update.get('sigma', self.config.get('sigma', 0.05))
        alpha = update.get('lr', self.config.get('lr', 0.01))

        optimizer = optimizer or self.optimizer
        if optimizer == None:
            raise Exception("update_from_epoch requires optimizer")

        for seed in seed_dict:
            gen = torch._C.Generator(device=self.device)
            gen.manual_seed(abs(int(seed)))
            psign = self.sign(int(seed))

            partial_delta = torch.randn((self.NPARAMS,), generator=gen, requires_grad=False,
                                 device=self.device) * (sigma * seed_dict[seed] * psign)

            optimizer.process_subgrad(partial_delta)
            delta += partial_delta

        if optimizer.set_params:
            self.set_from_delta(optimizer.compute_grads(delta, alpha))
        else:
            delta = optimizer.compute_grads(delta, alpha)
            self.update_from_delta(delta)
            # print(torch.std(delta), torch.mean(delta), torch.median(delta), torch.max(delta))
        return delta

    def apply_seed(self, seed, scalar=1., sigma=None):
        gen = torch._C.Generator(device=self.device)
        seed = int(seed)
        gen.manual_seed(abs(seed))
        delta = torch.randn((self.NPARAMS,), generator=gen, requires_grad=False,
                            device=self.device) * ((sigma or self.config['sigma']) * scalar * self.sign(seed))
        self.update_from_delta(delta)

    def apply_sparse_delta(self, sparse_params):
        idx = 0
        i = 0
        for param in self.model.parameters():
            size = np.product(self.model_shapes[i])
            block = delta[idx:idx+size]
            block = block.view(self.model_shapes[i])
            if sparse_params.get(i, False):
                param.data += sparse_params[i]
            i += 1
            idx += size

    def set_config(self, config):
        self.config = config

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def reset(self):
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def jit(self):
        self.model = torch.jit.script(self.model)

    # Used to check from floating point errors
    def get_checksum(self):
        acc = 0.
        params = self.model.parameters()
        for param in params:
            acc += float(torch.sum(param.data)) / len(self.model_shapes)
            acc += float(torch.mean(param.data)) + float(torch.max(param.data))

        return acc


class ESWorker(DistributedWorker):
    def __init__(self, pipe):
        super().__init__(pipe)
        self.results = {}
        self.cseeds = []
        self.model = None
        self.env = None
        self.run_config = {}

    def loop(self):
        cidx = len(self.results)
        if cidx < len(self.cseeds):
            # Apply model changes
            # start_time = time.time()

            cmodel = copy.deepcopy(self.model)
            cmodel.apply_seed(self.cseeds[cidx], scalar=1.)
            # cmodel.jit()
            episode_reward = 0.
            rstate = np.random.RandomState(self.env_seed)  # Is mutated by env

            # Accumulate rewards accross different seeds (virtual batch)
            for x in range(self.env_episodes):
                self.env.randomize(rstate)
                state = self.env.reset()
                cmodel.reset()
                
                # TODO: model_autoregressive batch optimization
                # Run through simulation
                done = False
                while not done:
                    with torch.no_grad():
                        action = cmodel(state.to(self.init_config['device']))

                    # Make no assumption on the format of action (no .cpu())
                    state, reward, done = self.env.step(action)
                    episode_reward += reward
                # print(action)

            # print('finished eval in %.3f' % (time.time() - start_time))

            # Average reward over all runs
            self.results[self.cseeds[cidx]] = episode_reward / self.env_episodes

            # Revert model changes
            # self.model.apply_seed(self.cseeds[cidx], scalar=-1.)
        elif len(self.results) == len(self.cseeds) and len(self.results) > 0:
            # print('sending %d/%d rewards back' % (cidx, len(self.cseeds)))
            # print(self.results)
            # print('checksum after:', self.model.get_checksum())
            self.send({'rewards': self.results})
            self.results = {}
            self.cseeds = []

    def create_model(self):
        torch.manual_seed(2)
        self._model = self.init_config['model_class'](
            *self.init_config['model_args'], **self.init_config['model_kwargs'])
        self.model = ModelWrapper(self._model, device=self.init_config['device'])
        self.model.set_config(self.run_config)
        self.optimizer = self.init_config['optimizer']
        self.optimizer.reset(self.model.NPARAMS, torch.nn.utils.parameters_to_vector(self.model.model.parameters()).detach())
        
    def create_env(self):
        self.env_class = self.init_config['env_class']
        self.env_config = self.init_config['env_config']
        self.env_episodes = self.init_config['env_episodes']
        self.env = self.env_class(**self.env_config)

    def handle_msg(self, msg):
        if type(msg) == dict:
            # print(msg)
            if msg.get('init', False):
                self.init_config = msg['init']
                if self.init_config['device'] == 'cuda':
                    self.init_config['device'] += ':%d' % np.random.randint(0, torch.cuda.device_count())
                self.create_model()
                self.create_env()

            if msg.get('run', False):
                self.cseeds = msg['run']['seeds']
                self.env_seed = msg['run']['env_seed']
                self.run_config = msg['run']['config']
                self.model.set_config(self.run_config)
                self.results = {}
                # print('Running', self.env_seed)

            if msg.get('update_history', False):
                # Recreate model
                for update in msg['update_history']:
                    self.model.update_from_epoch(update, optimizer=self.optimizer)

            if msg.get('update', False):
                # {seed: weight, ...}
                self.model.update_from_epoch(msg['update'], optimizer=self.optimizer)
                # TODO: auto validation?
                # print('checksum:', self.model.get_checksum())


class ESManager(DistributedManager):
    def __init__(self, config):
        super().__init__(("0.0.0.0", 8003,))
        self.tasked = set()
        self.done = False
        self.config = config
        self.iconfig = copy.deepcopy(config)
        
        if 'cuda' in self['device']:
            # For pytorch CUDA
            from multiprocessing import set_start_method
            set_start_method('spawn')

        self.results = {}
        self.update_history = []
        self.raw_history = []
        self.config_history = []
        self.epoch = 0
        self.writer = SummaryWriter('runs/%s' % self['logdir'])
        self.last_print = None
        self.last_epoch = 0
        self.epoch_ttl = 60 * 5

        self.env = self['env_class'](**self['env_config'])

        self.rand = np.random.RandomState(seed=self.config.get('seed', 42))
        # Seperated for reproducibility across populations
        self.env_rand = np.random.RandomState(seed=self.config.get('seed', 42)) 

        torch.manual_seed(2)
        _model = self['model_class'](
            *self['model_args'], **self['model_kwargs'])
        self.model = ModelWrapper(_model, device=self['device'])
        self.model.set_config({'sigma': self['sigma'], 'lr': self['lr']})

        self['optimizer'].reset(self.model.NPARAMS, torch.nn.utils.parameters_to_vector(self.model.model.parameters()))

    def stop(self):
        super().stop()
        self.listener.close()

    def __getitem__(self, k):
        return self.config[k]

    def print_log(self, delta, rewards, log_stats=False):
        # delta_e = np.sqrt(float(torch.sum(delta ** 2)))
        _r = list(rewards.values())
        r = np.mean(_r)
        r_sigma = np.std(_r)
        epoch_s = time.time() - self.last_print if self.last_print else 0.
        # Should be < 1, if > 1 model will adjust further than any of the tests did
        delta_e = torch.norm(delta)
        print("[Epoch %d, %.1fs/e]:\t mΔe: %.4f, lr: %.3f, σ: %.3f, ⟨r⟩: %.1f, rσ: %.2f, p: %d, workers: %d" %
              (self.epoch, epoch_s, delta_e, self['lr'], self['sigma'], r, r_sigma, len(rewards), len(self.get_active_workers())))
        self.last_print = time.time()
        # self.writer.add_scalar('delta_e', delta_e, self.epoch)
        self.writer.add_scalar('pop', len(rewards), self.epoch)
        self.writer.add_scalar('avg_reward', r, self.epoch)
        self.writer.add_scalar('std_reward', r_sigma, self.epoch)
        self.writer.add_scalar('lr', self['lr'], self.epoch)
        self.writer.add_scalar('sigma', self['sigma'], self.epoch)
        self.writer.add_scalar('update_norm', delta_e, self.epoch)
        if log_stats:
            print(torch.std(delta), torch.mean(delta),
                  torch.median(delta), torch.max(delta))
            # print(rewards)

    def get_processed_rewards(self):
        x = np.array(list(self.results.values()))

        if self['reward_norm'] == 'ranked':
            x = compute_centered_ranks(x)
        elif self['reward_norm'] == 'stdmean':
            mean_reward = np.mean(x)
            std_reward = np.std(x)
            x = ((x - mean_reward) / std_reward)
            x /= x.max()
        else:
            # No normalization
            pass

        # TODO: validate
        if x.std() == 0.:
            i = self.rand.randint(0, len(fresult))
            x[:] = 0
            x[i] += 1 # Choose random agent to be the new leader

        rewards = {}
        for k, r in zip(self.results.keys(), list(x)):
            rewards[k] = r
        #! DEBUG
        return rewards

    def get_update(self):
        fresult = self.get_processed_rewards()
        update = {'sigma': self['sigma'], 'lr': self['lr'], 'deltas': fresult}
        return update

    def eval(self):
        model = copy.deepcopy(self.model.model)

        self.env.randomize(np.random.RandomState(420691))
        state = self.env.reset()
        model.reset()

        # Run through simulation
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                action = model(state.to(self['device']))

            # Make no assumption on the format of action
            state, reward, done = self.env.step(action)
            episode_reward += reward

        self.writer.add_scalar('master_reward', episode_reward, self.epoch)

        data = self.env.eval()
        for key in data.get('scalars', {}):
            self.writer.add_scalar(key, data['scalars'][key], self.epoch)

        for key in data.get('images', {}):
            self.writer.add_image(key, data['images'][key], self.epoch)

    def on_es_ready(self):
        self.running = False
        update = self.get_update()

        self.raw_history.append(self.results)
        self.update_history.append(update)

        delta = self.model.update_from_epoch(update, optimizer=self['optimizer'])
        self.print_log(delta, self.results)
        self.broadcast({'update': update})
        self.results = {}
        self.epoch += 1

        if self.epoch % self['env_eval_every'] == 0:
            self.eval()

    def loop_es_busy(self):
        pass

    def scheduler_step(self):
        self.config['lr'] *= self['lr_decay']
        self.config['sigma'] *= self['sigma_decay']

    def loop(self):
        # No tasks pending, wait on exit
        if self.epoch > self['epochs']:
            self.done = True
            return

        # Got all results
        if len(self.tasked) == 0 and len(self.results) > 0:
            self.on_es_ready()
            return

        if self.last_epoch + self.epoch_ttl < time.time():
            print('Timeout, uncompleted tasks (%d/%d):' % (len(self.results), self['popsize']) , self.tasked)
            self.tasked = set()

        # Awaiting Results
        if len(self.tasked) > 0:
            self.loop_es_busy()
            return

        # Send tasks to workers
        self.last_epoch = time.time()
        self.running = True
        tasks = self.rand.randint(0, 2**31, size=(self['popsize'],))
        if self['antithetic']:
            tasks = np.hstack((tasks, -tasks))

        if self.epoch > self['pre_training_epochs']:
            env_seed = self.env_rand.randint(0, 2**31)
        else:
            env_seed = 420691

        self.scheduler_step()

        active = self.get_active_workers()
        if active:
            chunk = len(tasks) // len(active)
            for x in active:
                if active[-1] != x:
                    task = tasks[:chunk]
                    tasks = tasks[chunk:]
                else:
                    # Last node takes remaining tasks
                    task = tasks
                    tasks = []

                # print('%d: %d tasks' % (x, len(task)))

                self.send(x, {'run': {'seeds': task, 'env_seed': env_seed,
                                    'config': {'sigma': self['sigma'], 'lr': self['lr']}}})
                self.tasked.add(x)

    def on_new_worker(self, worker: int):
        print('New worker added %d' % worker)
        self.send(worker, {'init': self.iconfig,
                           'update_history': self.update_history})

    def on_worker_disconnect(self, worker: int):
        print('Worker disconnected %d' % worker)
        # Worker removed; this epoch will have slightly smaller population :/
        try:
            self.tasked.remove(worker)
        except KeyError:
            print('Tried removing %d from tasked but no was not assigned any task (or already completed the task)' % worker)

    def handle_msg(self, worker: int, msg: Any):
        # Worker finished it's task
        # print('got message', msg)
        if msg.get('rewards', False):
            # print('Finished worker %d' % worker)
            # TODO dynamic task allocation based on relative delivery time?
            for num in msg['rewards']:
                self.results[num] = msg['rewards'][num]
            self.tasked.remove(worker)
