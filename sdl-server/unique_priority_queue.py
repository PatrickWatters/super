from queue import Queue
import heapq
from  batch import BatchGroup
from threading import Thread

class UniquePriorityQueue(Queue):

    def _init(self, maxsize):
        self.queue = []
        self.REMOVED = object()
        self.entry_finder = {}
        self.batch_groups:dict[str, BatchGroup] = None
        self.consumers = []

    def _put(self, item, heappush=heapq.heappush):
        item = list(item)
        priority, task = item
        job_id, group_id, batch_id,action = task
        if batch_id in self.entry_finder:
            previous_item = self.entry_finder[batch_id]
            previous_job_id = previous_item[1][0]
            previous_priority, _ = previous_item

            if priority > previous_priority and previous_job_id == job_id: #if same job  always update the access time!
                previous_item[-1] = self.REMOVED
                self.entry_finder[batch_id] = item
                heappush(self.queue, item)
                print('batch_queud', batch_id)

            elif priority < previous_priority: #if a different job but a higher priorty access time - update!
                # Remove previous item
                previous_item[-1] = self.REMOVED
                self.entry_finder[batch_id] = item
                heappush(self.queue, item)
            else:
                # Do not add new item.
                #print('duplicate entry with lower prioirty so not added',item)
                pass
        else:
            self.entry_finder[batch_id] = item
            heappush(self.queue, item)
            print('batch_queud', batch_id)

    def _qsize(self, len=len):
        return len(self.entry_finder)
    
    def _get(self, heappop=heapq.heappop):
        while self.queue:
            item = heappop(self.queue)
            _, task = item
            if task is not self.REMOVED:
                job_id, group_id, batch_id, action = task
                del self.entry_finder[batch_id]
                return item
        raise KeyError('It should never happen: pop from an empty priority queue')
    
    def set_groups(self, g):
        self.batch_groups = g

    def start_consumers(self, num_consumers=32):
        # create a shared lock
        for i in range(num_consumers):
            name = 'Consumer-{}'.format(i)
            c = Thread(name=name,target=self.process_item, args=())
            c.start()
            self.consumers.append(c)

    def process_item(self,):
        while True:
            if not self.empty():
                priority, item = self.get()
                job_id, group_id, batch_id,action = item
                batch_group:BatchGroup = self.batch_groups[group_id]
                
                if action == 'prefetch':
                    batch_group.fetch_batch_via_lambda(batch_id, 
                                            include_batch_data_in_response=False,
                                            isPrefetch=True)
                if action == 'ping':
                    batch_group.keep_alive_batch_ping(batch_id, prefetch_on_cache_miss=True)