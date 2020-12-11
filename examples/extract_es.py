from distributed_worker import DistributedWorker
from portable_es import *
from tqdm import tqdm
import multiprocessing
import time
import torch

class ESExtractor(DistributedWorker):
    def __init__(self, pipe):
        super().__init__(pipe)
        self.results = {}
        self.cseeds = []
        self.model = None
        self.env = None
        self.run_config = {}

    def loop(self):
        pass

    def create_model(self):
        torch.manual_seed(2)
        self._model = self.init_config['model_class'](
            *self.init_config['model_args'], **self.init_config['model_kwargs'])
        self.model = ModelWrapper(self._model)

        self.optimizer = copy.deepcopy(self.init_config['optimizer'])
        self.optimizer.reset(self.model.NPARAMS, torch.nn.utils.parameters_to_vector(self.model.model.parameters()))

    def handle_msg(self, msg):
        if type(msg) == dict:
            # print(msg)
            if msg.get('init', False):
                self.init_config = msg['init']
                self.create_model()

            if msg.get('update_history', False):
                # Recreate model
                for update in tqdm(msg['update_history']):
                    self.model.update_from_epoch(update, optimizer=self.optimizer)

client_args = (('localhost', 6000), 'AF_INET', b'secret password')
pipe = multiprocessing.connection.Client(*client_args)
worker = ESExtractor(pipe)
while worker.model == None:
  worker.run_once()
torch.save(worker.model.model, 'es-extract.pt')
