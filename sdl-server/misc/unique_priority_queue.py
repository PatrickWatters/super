from queue import Queue
import heapq
import numpy as np
import threading
import time
import logging
import random
from concurrent.futures import ThreadPoolExecutor


class UniquePriorityQueue(Queue):

    def _init(self, maxsize):
        self.queue = []
        self.REMOVED = object()
        self.entry_finder = {}

    def _put(self, item, heappush=heapq.heappush):
        item = list(item)
        priority, task = item
        
        if task in self.entry_finder:
            previous_item = self.entry_finder[task]
            previous_priority, _ = previous_item
            if priority < previous_priority:
                # Remove previous item
                previous_item[-1] = self.REMOVED
                self.entry_finder[task] = item
                heappush(self.queue, item)
            else:
                # Do not add new item.
                print('duplicate entry with lower prioirty so not added',item)
                pass
        else:
            self.entry_finder[task] = item
            heappush(self.queue, item)
    def _qsize(self, len=len):
        return len(self.entry_finder)
    
    def _get(self, heappop=heapq.heappop):
        while self.queue:
            item = heappop(self.queue)
            _, task = item
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return item
        raise KeyError('It should never happen: pop from an empty priority queue')



q = UniquePriorityQueue()
for i in range(0, 100):
    q.put((np.Infinity,i))



class ConsumerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        #while True:
            while not q.empty():
                item = q.get()
                print(self.getName() + ' getting ' + str(item)  + ' : ' + str(q.qsize()) + ' items in queue')
                time.sleep(random.random())
            return
    
if __name__ == '__main__':
    consumers = []

    end = time.time()
    for i in range(8):
        name = 'Consumer-{}'.format(i)
        c = ConsumerThread(name=name)
        c.start()
        consumers.append(c)
    
    for consumer in consumers:
        consumer.join()

    print('ended',time.time() - end)                

    '''
    queue = UniquePriorityQueue()
    queue.put((4,'b1'))
    queue.put((3,'b3'))
    queue.put((3,'b2')) 
    queue.put((1,'b1'))
    '''
  