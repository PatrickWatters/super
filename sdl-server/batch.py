import numpy as np
from collections import OrderedDict
from lambda_wrapper import LambdaWrapper
import json
from misc.redis_client import RedisClient
import time
from threading import Lock


class Batch():
    def __init__(self,id, group_id,indices,labelled_paths):
        self.batch_id = id
        self.indices = indices
        self.last_pinged_timestamp = None
        self.processed_by =[]
        self.isCached = False
        self.group_id = group_id
        self.labelled_paths = labelled_paths
        self.size = len(self.indices)
        self.inProgress = False

class BatchGroup():
    def __init__(self,group_id, batches, lambda_wrapper:LambdaWrapper, redis_client:RedisClient, use_lambda:bool = True):
        self.group_id = group_id
        self.batches:dict[str,Batch] = batches
        #self.priorityq = UniquePriorityQueue()
        self.processed_by =[]
        self.isActive = True
        self.lambda_wrapper:LambdaWrapper =lambda_wrapper
        self.redis_client:RedisClient = redis_client
        self.lock = Lock()
        self.use_lambda = use_lambda

    def batchIsInProgress(self,bacth_id):
        return self.batches[bacth_id].inProgress
    
    def setBatchIsInProgress(self,bacth_id, progress_status):
        with self.lock:
            self.batches[bacth_id].inProgress = progress_status

    def batchIsCached(self,bacth_id):
        return self.batches[bacth_id].isCached
    
    def setCachedSatus(self,bacth_id, cached_status):
        with self.lock:
             self.batches[bacth_id].isCached = cached_status
    
    def setProcessedBy(self,bacth_id,job_id):
        with self.lock:
            self.batches[bacth_id].processed_by.append(job_id)
    
    def getlastPingedTimestamp(self,bacth_id):
       return self.batches[bacth_id].last_pinged_timestamp
    
    def setlastPingedTimestamp(self,bacth_id):
        with self.lock:
            self.batches[bacth_id].last_pinged_timestamp = time.time()
    
    def get_batch_ids(self):
          return list(self.batches.keys())
    
    def get_batch_indices(self,batch_id):
        return self.batches[batch_id].indices
    
    def fetch_batch_via_cache(self, batch_id):
        batch_data = self.redis_client.get_data(batch_id)
        
        if batch_data is not None:
            self.setlastPingedTimestamp(batch_id)
            cache_hit = True
            return batch_data, cache_hit
        else:
            self.setCachedSatus(batch_id, False)
            cache_hit = False
            return batch_data, cache_hit
    
    def fetch_batch_via_lambda(self, batch_id,include_batch_data_in_response = True, isPrefetch = False):

        if isPrefetch:
            cache_after_retrevial = True
        else:
            cache_after_retrevial = True

        if self.use_lambda:
            response = self.lambda_wrapper.invoke_function(labelled_paths=self.batches[batch_id].labelled_paths,
                                                           batch_id=batch_id,
                                                           cache_after_retrevial=cache_after_retrevial,
                                                           include_batch_data_in_response= include_batch_data_in_response,
                                                           get_log= False)
        else:
            response = self.lambda_wrapper.fetch_from_local_disk(labelled_paths=self.batches[batch_id].labelled_paths,
                                                           batch_id=batch_id,
                                                           cache_after_retrevial=cache_after_retrevial,
                                                           include_batch_data_in_response= include_batch_data_in_response,
                                                           get_log= False, redis_client=self.redis_client)
        paylaod = json.loads(response['Payload'])
        if 'errorMessage' in paylaod:
            print(paylaod['errorMessage'])
        
        elif paylaod['isCached'] == True:
            self.setCachedSatus(batch_id, True)
            self.setlastPingedTimestamp(batch_id)

        return paylaod['batch_data']
    

    def prefetch_batch(self, batch_id):
        if self.batchIsCached(batch_id) or self.batchIsInProgress(batch_id):
            return
                    
        self.setBatchIsInProgress(batch_id, True)

        self.fetch_batch_via_lambda(batch_id=batch_id,
                                    include_batch_data_in_response = False,
                                    isPrefetch= True )
        
        self.setBatchIsInProgress(batch_id, False)
    
    
    def keep_alive_batch_ping(self, batch_id, prefetch_on_cache_miss = False):
        response = self.redis_client.get_data(batch_id)
        if response is not None:
            self.setlastPingedTimestamp(batch_id)
        else:
            #cache miss - data must have been evicted by AWS
            self.setCachedSatus(batch_id, False)
            if prefetch_on_cache_miss:
                self.prefetch_batch(batch_id=batch_id)
    
    

    def find_subsitution_batch(self,job_id,next_batch_id):
        response = None
        cache_hit = False
        for batch_id in self.batches:
            if self.batches[batch_id].isCached and job_id not in self.batches[batch_id].processed_by:
                response = self.fetch_data_via_cache(batch_id)
                if response is None:
                    #batch incorrectly labelled as cached
                    self.setCachedSatus(batch_id, False)
                else:
                    cache_hit = True
                    next_batch_id = batch_id
                    break
        return next_batch_id, response,cache_hit