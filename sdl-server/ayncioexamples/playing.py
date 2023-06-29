from torch.utils.data.sampler import RandomSampler,BatchSampler, SequentialSampler
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

prepared_batches = queue.Queue(maxsize=25)
executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='job {}'.format(2))

def gen_sampler():
    xs = list(range(4))
    dataset = list(zip(xs,xs))
    
    base_sampler = RandomSampler(dataset)
    
    batch_sampler = BatchSampler(base_sampler, batch_size=2, drop_last=False)
    return batch_sampler

def start_consumer():
    global prepared_batches
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

    daemon = threading.Thread(target=start_consumer, daemon=True, name='Monitor')
    daemon.start()

    while True:
        fetch_batch()

