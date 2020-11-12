import math
from collections import namedtuple
from .utils import initializer

# TODO: check for lr/sigma in portable_es with check for callable/non-callable
StaticSchedule = namedtuple('StaticSchedule', 'lr, sigma')

class DecayScheduler:
    @initializer
    def __init__(self, ilr, isigma, lr_decay=0.9995, sigma_decay=0.9995):
        pass

    def step(self):
        self.ilr *= self.lr_decay
        self.isigma *= self.sigma_decay

    def lr(self):
        return self.ilr

    def sigma(self):
        return self.isigma


class CosineScheduler:
    @initializer
    def __init__(self, lr_min, lr_max, sigma_min, sigma_max, period = 50, warmup = 50, lr_decay = 0.999):
        self.step += 1
        self.tt = 0.
        self.dt = math.pi/float(2.0*self.period)

    def step(self, step: int):
        self.step += 1
        self.tt = self.dt * (self.step % int(self.period))
        self.lr_min = self.lr_min * self.lr_decay
        self.lr_max = self.lr_max * self.lr_decay
        
    def lr(self):
        if self.step < self.warmup:
            return self.lr_max
        else:
            return self.lr_min + 0.5*(self.lr_max - self.lr_min)*(1 + math.cos(self.tt))

    def sigma(self):
        if self.step < self.warmup:
            return self.sigma_max
        else:
            return self.sigma_min + 0.5*(self.sigma_max - self.sigma_min)*(1 + math.cos(self.tt))
  