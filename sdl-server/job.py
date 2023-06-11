from collections import OrderedDict
from batch import Batch, BatchGroup
import time
import logging
import datetime
from unique_priority_queue import UniquePriorityQueue
from misc.repeated_timer import RepeatedTimer
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

class MLTrainingJob():
    def __init__(self,job_id,batch_size,look_ahead_distance,warm_up_distance,global_priority_queue,use_substitutional_hits):
        self.job_id = job_id
        self.batch_size = batch_size
        self.warm_up_distance = warm_up_distance
        self.look_ahead_distance=look_ahead_distance
        self.use_substitutional_hits = use_substitutional_hits
        self.avg_training_speed = 0
        self.data_laoding_delay = 0
        self.total_epochs_processed = -1
        self.total_batches_processed = -1
        self.job_started_timestamp = None
        self.job_finished_timestamp = None
        self.global_priority_queue:UniquePriorityQueue = global_priority_queue
        self.warm_up_over = False
        self.job_timer = None
        self.executor = ThreadPoolExecutor() # create a thread pool with the default number of worker threads
        #self.data_prep_service = RepeatedTimer(0.012,self.run_batch_access_time_prediction)
        
        #epoch level attributes
        self.currEpoch_batchesProcessed = -1
        self.currEpoch_batchGroup:BatchGroup = None
        self.currEpoch_timer = None
        #self.currEpoch_batches = None
        self.currEpoch_remainingBatches = []

    def run_batch_access_time_prediction(self):
        predicted_times ={}
        for idx in range(0,min(len(self.currEpoch_remainingBatches),self.look_ahead_distance)):
            batch_id = self.currEpoch_remainingBatches[idx]
            #predicted_access_time = max(0,(((self.currEpoch_batchesProcessed+(idx-1)) * self.avg_training_speed) + self.data_laoding_delay) - (time.time() - self.currEpoch_timer)) #be careful with parentheness here!
            #predicted_times[batch_id] = predicted_access_time
            predicted_access_time = (((self.currEpoch_batchesProcessed+(idx-1)) * self.avg_training_speed) + self.data_laoding_delay) - (time.time() - self.currEpoch_timer) #be careful with parentheness here!

            if not self.currEpoch_batchGroup.batchIsCached(batch_id) and not self.currEpoch_batchGroup.batchIsInProgress(batch_id) :
                #batch is not cached and is coming soon, so prefetch it
                self.global_priority_queue.put((predicted_access_time,(self.job_id, self.currEpoch_batchGroup.group_id, batch_id, 'prefetch')))
                #self.global_priority_queue.put((self.job_id, self.currEpoch_batchGroup.group_id, batch_id, 'prefetch'))

            #else:
            #    last_pinged = self.currEpoch_batchGroup.getlastPingedTimestamp(batch_id)

            #    if last_pinged is None or time.time() - last_pinged > (60 * 10): #if it batch hasn't been pinged in over 10mins
            #        self.global_priority_queue.put((predicted_access_time,(self.job_id, self.currEpoch_batchGroup.group_id, batch_id, 'ping')))

        #self.global_priority_queue.put((predicted_access_time,(self.job_id, self.currEpoch_batchGroup_id, batch_id)))
        #logging.info((time.time() - self.currEpoch_timer,self.data_laoding_delay))
        #logging.info(predicted_times)

    def increment_epochs_processed(self):
        self.total_epochs_processed +=1
        if self.total_epochs_processed ==0:
            self.job_started_timestamp = datetime.datetime.now()
            self.job_timer = time.time()

    def increment_batches_processed(self):
        self.total_batches_processed +=1
        self.currEpoch_batchesProcessed +=1

        if self.total_batches_processed == self.warm_up_distance: 
            self.reset_epoch_timer()
            self.reset_dl_delay()
            self.warm_up_over = True
            #self.data_prep_service.start()

    def set_batches_for_new_epoch(self,batch_group: BatchGroup):
        self.currEpoch_batchGroup = batch_group
        self.currEpoch_remainingBatches=batch_group.get_batch_ids()
        self.currEpoch_batchesProcessed =0



    def set_avg_training_speed(self,speed):
        self.avg_training_speed = speed
    
    def update_data_laoding_delay(self,delay):
        self.data_laoding_delay += delay
    
    def reset_dl_delay(self):
        self.data_laoding_delay = 0
    
    def reset_epoch_timer(self):
        self.currEpoch_timer = time.time()

    def end_job(self):
        #self.data_prep_service.stop()
        self.job_finished_timestamp = datetime.datetime.now()
        duration =  time.time() - self.job_timer
        self.active = False
        return self.job_started_timestamp, self.job_finished_timestamp, duration

    def find_next_batch(self):
        
        next_batch_id = self.currEpoch_remainingBatches[0]
        isCached = self.currEpoch_batchGroup.batchIsCached(next_batch_id)

        batch_data = None
        cache_hit = False

        if isCached:
            batch_data,cache_hit  = self.currEpoch_batchGroup.fetch_data_via_cache(next_batch_id)
        
        if batch_data is None and self.use_substitutional_hits:
            next_batch_id, batch_data,cache_hit = self.currEpoch_batchGroup.find_subsitution_batch(self.job_id, next_batch_id)
        
        if batch_data is None:
            batch_data = self.currEpoch_batchGroup.fetch_data_via_s3(next_batch_id)
            cache_hit = False
        
        self.currEpoch_remainingBatches.remove(next_batch_id)
        self.currEpoch_batchGroup.setProcessedBy(next_batch_id, self.job_id)

        batch_incides = self.currEpoch_batchGroup.get_batch_indices(next_batch_id)

        self.increment_batches_processed()
               
        if self.warm_up_over:
            #self.executor.submit(self.run_batch_access_time_prediction) # does not block
            self.run_batch_access_time_prediction()

        return next_batch_id, cache_hit, batch_data, batch_incides

        
    def _find_substitute_batch(self,next_batch_id, next_batch_indices):
        return next_batch_id, next_batch_indices, False