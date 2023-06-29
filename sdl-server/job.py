from threading import Thread
import queue
from coordinator import DataFeedCoordinator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from misc.redis_client import RedisClient
from misc.lambda_wrapper import LambdaWrapper
import json
import time
import logging
import sys
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class MLTrainingJob():
    def __init__(self,job_id,coordinator:DataFeedCoordinator, args):
        self.job_id = job_id
        self.total_epochs_processed = 0
        self.total_batches_processed = 0
        self.job_started_timestamp = None
        self.job_finished_timestamp = None
        self.coordinator = coordinator
        self.maxworkers = 30
        self.prepared_batches = queue.Queue(maxsize=25)
        self.job_end = False
        self.activeBatchSetId = None
        self.redis_client:RedisClient = RedisClient(args.redis_host,args.redis_port)
        self.lambda_wrapper = LambdaWrapper(args.s3_bucket, args.redis_host, args.redis_port,function_name=args.lambda_func_name)

    def fetch_batch_data(self, batchId):
        data = None
        if self.coordinator.batch_is_cached(self.activeBatchSetId, batchId):
            data = self.redis_client.get_batch(batchId)
            if data is not None:
                self.coordinator.update_batch_last_access_time(self.activeBatchSetId,batchId)
        
        if data is None and self.coordinator.batch_is_inProgress(self.activeBatchSetId,batchId):
            self.fetch_batch_data(batchId)
        
        if data is None:
            
            #not found in cache and not inProgress... lets do it the slower way
            self.coordinator.set_batch_inProgress(self.activeBatchSetId,batchId,True)
            #time.sleep(random.random())
            #fetch batch from lambda
            response = self.lambda_wrapper.invoke_function(
                labelled_paths=self.coordinator.get_batch_lablled_paths(self.activeBatchSetId,batchId),
                batch_id=batchId,
                cache_after_retrevial=False)
            
            paylaod = json.load(response['Payload'])

            if paylaod['isCached'] == True:
                self.coordinator.set_batch_isCached(self.activeBatchSetId,batchId,True)
                self.coordinator.update_batch_last_access_time(self.activeBatchSetId,batchId)
            data = paylaod['batch_data']
            #data = batchId
            self.coordinator.set_batch_inProgress(self.activeBatchSetId,batchId,False)
        
        return data,batchId

    def prep_batches(self):
        while not self.job_end: 
            batches,self.activeBatchSetId = self.coordinator.gen_new_bacthes_for_job(self.job_id,self.activeBatchSetId)
            with ThreadPoolExecutor(max_workers=self.maxworkers) as executor:
                
                #futures = [executor.submit(self.fetch_batch_data, i) for i in batches]
                 # process task results as they are available
                #for future in as_completed(futures):
                # retrieve the result
                #    result= future.result()
                #    self.prepared_batches.put(result)
                #    logger.info("batch {} queued".format(result))
                for result, batchid in executor.map(self.fetch_batch_data,batches):
                    self.prepared_batches.put((result, batchid))
                    logger.info("batch {} queued".format(batchid))

            self.total_epochs_processed +=1
            self.total_batches_processed +=len(batches)
    
    def next_batch(self):
        batch_data = self.prepared_batches.get()
        return batch_data[0], batch_data[1]
    
    def start_data_prep_workers(self):
        daemon = Thread(target=self.prep_batches, daemon=True, name='{} - Prep Daemon'.format(self.job_id))
        daemon.start()

    def handle_job_end(self):
         self.job_ended = True
         self.executor.shutdown()