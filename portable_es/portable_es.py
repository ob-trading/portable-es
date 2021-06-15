from distributed_worker import DistributedManager, DistributedWorker
from tensorboardX import SummaryWriter
from typing import Tuple, Any
from .timing import TimingManager
from .utils import params2vector, create_deco_meta
import torch
import copy
import pprint
import numpy as np
import time
import gc

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

def sign(a):
    return (a > 0) - (a < 0)

def get_random_tensor(shape, device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(abs(int(seed)))
    partial_delta = torch.randn((shape,), generator=gen, requires_grad=False, device=device).detach()
    return partial_delta

class ModelWrapper(metaclass=create_deco_meta([torch.no_grad()])):
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


    # Adds to params
    def update_from_delta(self, delta, alpha=1):
        delta = delta.detach()
        idx = 0
        i = 0
        for param in self.model.parameters():
            size = np.product(self.model_shapes[i])
            block = delta[idx:idx+size]
            block = block.view(self.model_shapes[i])
            i += 1
            idx += size
            param.data.add_(block, alpha=alpha)

    # Overwrites params
    def set_from_delta(self, delta):
        delta = delta.detach()
        idx = 0
        i = 0
        for param in self.model.parameters():
            size = np.product(self.model_shapes[i])
            block = delta[idx:idx+size]
            block = block.view(self.model_shapes[i])
            i += 1
            idx += size
            param.data.copy_(block)

    def update_from_epoch(self, update, log=False, optimizer=None):
        seed_dict = update['deltas']

        delta = torch.zeros(
            (self.NPARAMS,), requires_grad=False, device=self.device).detach()

        sigma = update['sigma']
        alpha = update['lr']

        optimizer = optimizer or self.optimizer
        if optimizer == None:
            raise Exception("update_from_epoch requires optimizer")

        for seed in seed_dict:
            psign = sign(int(seed))
            partial_delta = get_random_tensor(self.NPARAMS, str(self.device), seed) * (sigma * seed_dict[seed] * psign)

            optimizer.process_subgrad(partial_delta)
            delta += partial_delta

        if optimizer.set_params:
            self.set_from_delta(optimizer.compute_grads(delta, self.model, alpha))
        elif optimizer.internal_mut:
            # TODO: replace p_t_v with own impl for consistency
            x = params2vector(self.model.parameters()).detach()
            optimizer.compute_grads(delta, self.model, alpha)
            delta = params2vector(self.model.parameters()).detach() - x
        else:
            delta = optimizer.compute_grads(delta, self.model, alpha)
            self.update_from_delta(delta)
        
            # print(torch.std(delta), torch.mean(delta), torch.median(delta), torch.max(delta))
        return delta

    def apply_seed(self, seed, scalar=1., sigma=None):
        seed = int(seed)
        delta = get_random_tensor(self.NPARAMS, str(self.device), seed) 
        self.update_from_delta(delta, alpha=(sigma  * scalar * sign(seed)))

    def apply_sparse_delta(self, sparse_params):
        sparse_params = sparse_params.detach()
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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


class ESWorker(DistributedWorker, metaclass=create_deco_meta([torch.no_grad()])):
    def __init__(self, pipe):
        super().__init__(pipe)
        self.results = {}
        self.cseeds = []
        self.model = None
        self.env = None
        self.run_config = {}


    def loop(self):
        gc.collect()
        cidx = len(self.results)
        if cidx < len(self.cseeds):
            # Apply model changes
            with self.profiler.add('outer'): 
                with self.profiler.add('model-init'): 
                    cmodel = copy.deepcopy(self.model)
                    #cmodel = self.model
                    cmodel.apply_seed(self.cseeds[cidx], scalar=1., sigma=self.run_config['sigma'])

                # cmodel.jit()
                episode_reward = 0.
                rstate = np.random.RandomState(self.env_seed)  # Is mutated by env

                # Accumulate rewards accross different seeds (virtual batch)
                for x in range(self.env_episodes):
                    with self.profiler.add('env-init'):
                        self.env.randomize(rstate)
                        state = self.env.reset()
                        cmodel.reset()
                    
                    # TODO: model_autoregressive batch optimization
                    # Run through simulation
                    done = False
                    while not done:
                        with self.profiler.add('model-eval'):                    
                            action = cmodel.forward(state.to(self.init_config['device']))

                        # Make no assumption on the format of action (no .cpu())
                        with self.profiler.add('env-eval'):                    
                            state, reward, done = self.env.step(action)
                            episode_reward += reward
                        
                    # print(action)

                    # print('finished eval in %.3f' % (time.time() - start_time))

                    # Average reward over all runs
                    self.results[self.cseeds[cidx]] = episode_reward / self.env_episodes

            # Revert model changes
            # self.model.apply_seed(self.cseeds[cidx], scalar=-1.)
            #cmodel.apply_seed(self.cseeds[cidx], scalar=-1., sigma=self.run_config['sigma'])
            del cmodel
        elif len(self.results) == len(self.cseeds) and len(self.results) > 0:
            # print('sending %d/%d rewards back' % (cidx, len(self.cseeds)))
            # print(self.results)
            # print('checksum after:', self.model.get_checksum())
            # TODO: send back avg time so manager can estimate workload
            self.send({'rewards': self.results})
            self.results = {}
            self.cseeds = []
    
    def create_model(self):
        torch.manual_seed(2)
        self._model = self.init_config['model_class'](
            *self.init_config['model_args'], **self.init_config['model_kwargs'])
        self.model = ModelWrapper(self._model, device=self.init_config['device'])
        self.optimizer = self.init_config['optimizer']
        # NOTE: May hang on older version of pytorch for some reason, probably a deadlock in the cloning mechanism (<=1.4.0)
        # NOTE UPDATE: Seems to occur on 1.7.1 (alpine) as well, is fixed with OMP_NUM_THREADS=1, but only when set outside of the program :thinking:
        self.optimizer.reset(self.model.NPARAMS, params2vector(self.model.model.parameters()).detach(), self.model.model_shapes)
        
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
                self.results = {}
                self.profiler = TimingManager()
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


class ESManager(DistributedManager, metaclass=create_deco_meta([torch.no_grad()])):
    def __init__(self, config):
        super().__init__("0.0.0.0")
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
        self.last_epoch = 1e9
        self.epoch_ttl = 60 * 5

        self.env = self['env_class'](**self['env_config'])

        # Seperated for reproducibility across populations
        self.rand = np.random.RandomState(seed=self.config.get('seed', 42))
        self.env_rand = np.random.RandomState(seed=self.config.get('seed', 42)) 

        torch.manual_seed(2)
        _model = self['model_class'](
            *self['model_args'], **self['model_kwargs'])
        self.model = ModelWrapper(_model, device=self['device'])

        self['optimizer'].reset(self.model.NPARAMS, params2vector(self.model.model.parameters()).detach(), self.model.model_shapes)

    @property
    def _worker_config(self):
        return {'sigma': self.sigma, 'lr': self.lr}

    @property
    def lr(self):
        if callable(self['scheduler'].lr):
            return self['scheduler'].lr()
        else:
            return self['scheduler'].lr

    @property
    def sigma(self):
        if callable(self['scheduler'].sigma):
            return self['scheduler'].sigma()
        else:
            return self['scheduler'].sigma

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
              (self.epoch, epoch_s, delta_e, self.lr, self.sigma, r, r_sigma, len(rewards), len(self.get_active_workers())))
        self.last_print = time.time()
        # self.writer.add_scalar('delta_e', delta_e, self.epoch)
        self.writer.add_scalar('pop', len(rewards), self.epoch)
        self.writer.add_scalar('avg_reward', r, self.epoch)
        self.writer.add_scalar('std_reward', r_sigma, self.epoch)
        self.writer.add_scalar('lr', self.lr, self.epoch)
        self.writer.add_scalar('sigma', self.sigma, self.epoch)
        self.writer.add_scalar('update_norm', delta_e, self.epoch)
        if epoch_s > 0:
            self.writer.add_scalar('time_per_epoch', epoch_s, self.epoch)
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

        return rewards

    def get_update(self):
        fresult = self.get_processed_rewards()
        update = {**self._worker_config, 'deltas': fresult}
        return update

    def eval(self):
        self.env.randomize(np.random.RandomState(420691))
        state = self.env.reset()

        # Run through simulation
        done = False
        episode_reward = 0
        self.model.reset()
        while not done:
            action = self.model.forward(state.to(self['device']))

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
        if getattr(self['scheduler'], 'step', None):
            self['scheduler'].step()

        if self.epoch % self['env_eval_every'] == 0:
            self.eval()

    def loop_es_busy(self):
        pass

    def loop(self):
        gc.collect()
        # No tasks pending, wait on exit
        if self.epoch > self['epochs']:
            self.done = True
            return

        # Got all results
        if len(self.tasked) == 0 and len(self.results) > 0:
            self.on_es_ready()
            return

        # Awaiting Results
        if len(self.tasked) > 0:
            self.loop_es_busy()
            if self.last_epoch + self.epoch_ttl < time.time():
                print('Timeout, uncompleted tasks (%d/%d):' % (len(self.results), self['popsize']) , self.tasked)
                self.tasked = set()
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

        active = self.get_active_workers()
        if active:
            chunk = round(len(tasks) / len(active))
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
                                    'config': self._worker_config}})
                self.tasked.add(x)

    def on_new_worker(self, worker: int):
        print('New worker added %d' % worker)
        self.send(worker, {'init': self.iconfig,
                           'update_history': self.update_history})

    def on_worker_disconnect(self, worker: int):
        print('Worker disconnected %d' % worker)
        try:
            # Worker removed; this epoch will have slightly smaller population :/
            self.tasked.remove(worker)
        except KeyError:
            print('Tried removing %d from tasked but no was not assigned any task (or already completed the task)' % worker)

    def handle_msg(self, worker: int, msg: Any):
        # Worker finished it's task
        if msg.get('rewards', False):
            # print('Finished worker %d' % worker)
            # TODO dynamic task allocation based on relative delivery time?
            for num in msg['rewards']:
                self.results[num] = msg['rewards'][num]
            
            self.tasked.remove(worker)
