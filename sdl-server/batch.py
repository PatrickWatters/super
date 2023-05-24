from misc.priority_queue import PriorityQueue
import numpy as np
from collections import OrderedDict

class Batch():
    def __init__(self,id, group_id,indices):
        self.batch_id = id
        self.indices = indices
        self.last_pinged_timestamp = None
        self.processed_by =[]
        self.isCached = False
        self.isActive = True
        self.group_id = group_id
        self.size = len(self.indices)

class BatchGroup():
    def __init__(self,group_id, batches):
        self.group_id = group_id
        self.batches:dict = batches
        self.prefetch_priorityq = PriorityQueue()
        self.processed_by =[]
        self.cached_batches = []
        self.isActive = True

        #self.batch_ids = self.batches.keys()

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
