from misc.priority_queue import PriorityQueue
import numpy as np

class BatchGroup():
    def __init__(self,seed,set_of_batches):
        self.group_id = seed
        self.prefetch_priorityq = PriorityQueue()
        self.batches:dict[str, BatchMetadata] = {}
        for batch_id in set_of_batches:
            self.prefetch_priorityq.push(batch_id,np.Infinity)
            self.batches[batch_id] = BatchMetadata(id=batch_id,indices=set_of_batches[batch_id],group_id=self.group_id )
        self.processed_by =[]
        self.cached_batches = []
        self.isActive = True
        self.batches_dict = set_of_batches
    
    def update_batch_access_estimate(self,batch_id,access_time):
        self.prefetch_priorityq.push(batch_id, access_time)

    def findSubsituteBatch(self,candidate_batch_ids):
        overlap = set(candidate_batch_ids).intersection(self.cached_batches)
        if len(overlap) > 0:
            return overlap[0]
        else:
            return None
    
    def batchIsCached(self,bacth_id):
        return self.batches[bacth_id].isCached
    
    def getLastAccessed(self,bacth_id):
        return self.batches[bacth_id].last_accessed
    
    def getProcessedBy(self,bacth_id):
        return self.batches[bacth_id].processed_by
    
    def setCachedSatus(self,bacth_id, cached_status):
         self.batches[bacth_id].isCached = cached_status
    
    def setProcessedBy(self,bacth_id,job_id):
         self.batches[bacth_id].processed_by.append(job_id)
    
    def setLastAccessed(self,bacth_id,timestamp):
        self.batches[bacth_id].last_accessed = timestamp
        self.batches[bacth_id].last_pinged_timestamp = timestamp

    def setlastPingedTimestamp(self,bacth_id,timestamp):
        self.batches[bacth_id].last_pinged_timestamp = timestamp

    def setNextAccessTimeEstimate(self,bacth_id,access_time):
        if access_time < self.batches[bacth_id].next_access_time or self.batches[bacth_id].next_access_time is None:
            self.batches[bacth_id].next_access_time = access_time


class BatchMetadata():
    def __init__(self,id, indices, group_id):
        self.batch_id = id
        self.indices = indices
        self.last_pinged_timestamp = None
        self.processed_by =[]
        self.isCached = False
        self.isActive = True
        self.group_id = group_id