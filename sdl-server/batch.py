from misc.unique_priority_queue import UniquePriorityQueue
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
        self.priorityq = UniquePriorityQueue()
        self.processed_by =[]
        self.cached_batches = []
        self.isActive = True
        
            
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
