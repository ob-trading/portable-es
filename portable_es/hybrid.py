from .sac import ReplayMemory, SAC, QNetwork
from .portable_es import ESManager

sac_config = {
  'gamma': 0.99,
  'tau': 0.005,
  'lr': 0.0003,
  'alpha': 0.2,
  'policy': 'Guassian', # 'Guassian', 'Deterministic', torch.nn.Module
  'critic': QNetwork(num_inputs, outputs, hidden), # torch.nn.Module[state+action->(v1,v2,)]
  'target_update_interval': 1,
  'automatic_entropy_tuning': True,
  'action_space': (3,),
}

class ESACManager(ESManager):
    def __init__(self, config):
        super().__init__(config)
        self.sac_update = None
        self.sac_eval = None # Latch for SAC sampling/eval
        self.epsilon_start = 1.0
        self.epsilon_final = 0.1
        self.epsilon_decay = 1200000 / 5
        self.sac_interval = 5
        self.random_steps = 150
        self.max_sac_episodes = 5
        self.updates_per_step = 1
        self.sac_batch_size = 256
        self.memory = ReplayMemory()

    def epsilon_by_frame(self, frame_idx):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * frame_idx / self.epsilon_decay)

    def get_update(self):
        fresult = self.get_processed_rewards()
        update = {'sigma': self.sigma, 'lr': self.lr, 'deltas': fresult}
        if self.sac_update:
            update['abs'], _ = self.crossover(self.sac_update, self.model)
            self.sac_update = None
        self.sac_eval = None
        return update

    def crossover(self, a, b):


        return a_map, b_map

    def loop_es_busy(self):
        super().loop_es_busy()
        # TODO: ESAC here (awaiting results so update a secondary agent in mean time)
        # Latch sac_eval
        if self.epoch % self.sac_interval == 0 and not self.sac_update and self.sac_eval == None:
            self.sac_eval = self.rand.random() < self.epsilon_by_frame(self.epoch)

            if self.sac_eval == True:
                self.sac_update = copy.deepcopy(self.model)
                # TODO: Wrap SAC

                for i_episode in range(self.max_sac_episodes):
                    episode_reward = 0
                    episode_steps = 0
                    state = self.env.reset()

                    while not done:
                        if self.random_steps > self.epoch:
                            # Enviroment is expected to argmax/decode
                            action = self.rand.rand(*self.config['action_space'])
                        else:
                            action = self.sac_update.select_action(state) 

                        if len(self.memory) > self.sac_batch_size:
                            for i in range(self.updates_per_step):
                                critic_loss, qf_2_loss, policy_loss, ent_loss, alpha = self.sac_update.update_parameters(self.memory, self.sac_batch_size, updates)
                                updates += 1

                        next_state, reward, done = self.env.step(action)
                        episode_reward += reward
                        episode_steps += 1

                        # Ignore the "done" signal if it comes from hitting the time horizon.
                        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                        mask = float(not done)

                        self.memory.push(state, action, reward, copy.deepcopy(next_state), mask) # Append transition to memory

                        state = next_state