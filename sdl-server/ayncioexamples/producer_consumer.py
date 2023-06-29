from torch.utils.data.sampler import RandomSampler,BatchSampler, SequentialSampler
import queue
from time import sleep
from random import random
from threading import Thread
import asyncio

prepared_batches = queue.Queue(maxsize=25)

def gen_sampler():
    xs = list(range(4))
    dataset = list(zip(xs,xs))
    
    base_sampler = RandomSampler(dataset)
    
    batch_sampler = BatchSampler(base_sampler, batch_size=2, drop_last=False)
    return batch_sampler

def start_producer():
    
    while True:
        sampler  = gen_sampler()
        for i,batch_indiceis in enumerate(sampler):
            batch_id = abs(hash(frozenset(batch_indiceis)))
            prepared_batches.put(batch_id)
        #sampler = gen_sampler()

def fetch_batch():
  data = prepared_batches.get()
  print(data)

if __name__ == "__main__":
    daemon = Thread(target=start_producer, daemon=True, name='Monitor')
    daemon.start()
    print('Main thread is carrying on...')
    while True:
        fetch_batch()

