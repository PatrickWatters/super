from torch.utils.data.sampler import RandomSampler,BatchSampler, SequentialSampler
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from random import random
from time import sleep
import time

prepared_batches = queue.Queue(maxsize=25)
executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='job {}'.format(2))

def gen_sampler():
    xs = list(range(100))
    dataset = list(zip(xs,xs))
    
    base_sampler = RandomSampler(dataset)
    
    batch_sampler = BatchSampler(base_sampler, batch_size=2, drop_last=False)
    return xs

def process_bacth(batch_id):
    sleep(random())
    return f'Task: {batch_id} done.'

def start_consumer():
    global prepared_batches
    #while True:
    end = time.time()
    sampler  = gen_sampler()
    with ThreadPoolExecutor(10) as executor:
        for result in executor.map(process_bacth, sampler):
            #batch_id = abs(hash(frozenset(batch_indiceis)))
            prepared_batches.put(result)
        #sampler = gen_sampler()
    print(time.time() - end)


def fetch_batch():
  data = prepared_batches.get()
  print(data)

if __name__ == "__main__":
    daemon = threading.Thread(target=start_consumer, daemon=True, name='Monitor')
    daemon.start()

    while True:
        fetch_batch()

