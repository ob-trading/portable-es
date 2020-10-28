import torch

class GymWrapper:
    def __init__(self, gym_name='CartPole-v1', castf=lambda x: x, **kwargs):
        self.g = gym_name
        self.castf = castf

    def step(self, action) -> Tuple[torch.Tensor, float, bool]:
        obs, reward, done, info = self.env.step(self.castf(action))
        return torch.from_numpy(obs).float(), reward, done

    def randomize(self, np_randomstate):
        pass

    def reset(self):
        if not getattr(self, 'env', None):
            self.env = gym.make(self.g)
        return torch.from_numpy(self.env.reset()).float()

    def eval(self):
        return {}