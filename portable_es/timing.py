import time
import numpy as np
from dataclasses import dataclass

class Timing:
    """
    Calculates time it took to complete the marked block
    """

    def __init__(self):
        self.dtimes = []
  
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dtimes.append(time.time() - self.start_time)
      
    def summary(self):
        dtimes = np.array(self.dtimes)
        return {'mean': np.mean(dtimes), 'std': np.std(dtimes), 'count': len(dtimes), 'cumm': np.sum(dtimes)}

class TimingManager:
    """
    Manages different 'Timing's classes and creates an easily printable/picklable object
    """

    def __init__(self):
        self.timings = {}

    def add(self, name):
        if not self.timings.get(name, None):
            self.timings[name] = Timing()
        
        return self.timings[name]

    def summary(self):
        return dict([(n, self.timings[n].summary()) for n in self.timings])

