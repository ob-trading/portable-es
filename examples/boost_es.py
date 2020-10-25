from distributed_worker import create_remote_worker, create_remote_worker_sync
from portable_es import ESWorker

import os
import sys

abs_path = os.path.realpath('..')
if not abs_path in sys.path:
    sys.path.append(abs_path)
os.chdir('../')

client_args = (('localhost', 8003), 'AF_INET', b'secret password')


# create_remote_worker_sync(ESWorker, client_args)
# TODO: add threads to params
workers = []
for x in range(1):
  workers.append(create_remote_worker(ESWorker, client_args))


import time
time.sleep(500)
sys.exit()