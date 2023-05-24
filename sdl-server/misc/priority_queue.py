from heapq import heappush, heappop
import itertools
import threading
import numpy as np
__all__ = ["PriorityQueue"]

class PriorityQueue:
    sentinel = object()

    def __init__(self):
        self.heap, self.entries = [], {}
        self.counter = itertools.count()
        self.mutex = threading.Lock()    

    def push(self, item, priority):
        with self.mutex:
            if item not in self.entries:
                entry = self.entries[item] = [priority, next(self.counter), item]
                heappush(self.heap, entry)
            elif self.entries[item][0] > priority:
                self.entries[item][2] = self.sentinel
                entry = self.entries[item] = [priority, next(self.counter), item]
                heappush(self.heap, entry)

    def pop(self):
        with self.mutex:
            while True:
                priority, count, item = heappop(self.heap)
                if item is not self.sentinel:
                    del self.entries[item]
                    return item

if __name__ == '__main__':
    testq= PriorityQueue()
    testq.push('b1',np.Infinity)
    testq.push('b1',np.Infinity)
    testq.push('b2',np.Infinity)
    testq.push('b3',np.Infinity)
    testq.push('b3',4)


    print(testq.pop())
    print(testq.pop())
    print(testq.pop())
