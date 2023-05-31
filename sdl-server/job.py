from collections import OrderedDict
from batch import Batch
import time
import threading
import logging
import datetime
from misc.unique_priority_queue import UniquePriorityQueue
from misc.repeated_timer import RepeatedTimer

class MLTrainingJob():
    def __init__(self,job_id,batch_size,look_ahead_distance,warm_up_distance,access_time_update_freq,global_priority_queue):
        self.job_id = job_id
        self.batch_size = batch_size
        self.warm_up_distance = warm_up_distance
        self.look_ahead_distance=look_ahead_distance
        self.access_time_update_freq = access_time_update_freq
        self.avg_training_speed = 0
        self.data_laoding_delay = 0
        self.total_epochs_processed = -1
        self.total_batches_processed = -1
        self.current_epcoch_start_time = None
        self.current_batch_group = None
        self.job_started_timestamp = None
        self.job_finished_timestamp = None
        self.epoch_timer = None
        self.epoch_batches_remaining = []
        self.global_priority_queue:UniquePriorityQueue = global_priority_queue
        self.active = True
        self.data_prep_service = RepeatedTimer(self.access_time_update_freq ,self.run_data_prep_task)

    def run_data_prep_task(self):
        predicted_times ={}
        for idx in range(0,min(len(self.epoch_batches_remaining),self.look_ahead_distance)):
            batch_id = self.epoch_batches_remaining[idx+1]
            #predicted_access_time = max(0,(((self.total_batches_processed+(idx-1)) * self.avg_training_speed) + self.data_laoding_delay) - (time.time() - self.epoch_timer)) #be careful with parentheness here!
            predicted_access_time = max(0,(((self.total_batches_processed+(idx-1)) * self.avg_training_speed) + self.data_laoding_delay) - (time.time() - self.epoch_timer)) #be careful with parentheness here!

                        #predicted_access_time = max(0,((idx * self.avg_training_speed) + self.data_laoding_delay))
            #self.global_priority_queue.put((predicted_access_time,(self.job_id,self.current_batch_group,batch_id)))
            predicted_times[batch_id] = predicted_access_time
        logging.info((time.time() - self.epoch_timer,self.data_laoding_delay))
        logging.info(predicted_times)
        #logging.info(self.data_laoding_delay)

    def increment_epochs_processed(self):
        self.total_epochs_processed +=1
        if self.total_epochs_processed ==0:
            self.job_started_timestamp = datetime.datetime.now()

    def increment_batches_processed(self):
        self.total_batches_processed +=1
        if self.total_batches_processed == self.warm_up_distance: 
            self.reset_epoch_timer()
            self.reset_dl_delay()
            #self.data_prep_service.start()

    def set_batches_to_process(self,group_id, batches:dict[str,Batch]):
        self.current_epoch_batches = batches
        self.epoch_batches_remaining=list(batches.keys())
        self.current_batch_group = group_id

    def set_avg_training_speed(self,speed):
        self.avg_training_speed = speed
    
    def update_data_laoding_delay(self,delay):
        self.data_laoding_delay += delay
    
    def reset_dl_delay(self):
        self.data_laoding_delay = 0
    
    def reset_epoch_timer(self):
        self.epoch_timer = time.time()

    def end_job(self):
        self.data_prep_service.stop()
        self.job_finished_timestamp = datetime.datetime.now()
        self.active = False
        return self.job_started_timestamp, self.job_finished_timestamp

    def _next_batch(self,use_substitutional_hits):
        self.run_data_prep_task()

        isCached = False
        next_batch_id = self.epoch_batches_remaining[0]
        next_batch_indices = self.current_epoch_batches[next_batch_id].indices  
        if self.current_epoch_batches[next_batch_id].isCached:
            isCached = True
           # next_batch_indices = self.current_epoch_batches[next_batch_id].indices  
        if isCached == False and use_substitutional_hits:
            next_batch_id,next_batch_indices,isCached = self._find_substitute_batch(next_batch_id, next_batch_indices)
        
        self.epoch_batches_remaining.remove(next_batch_id)

        self.current_epoch_batches[next_batch_id].isCached = True #remove this line later - only added to check that the shllow copy with batch group is working
        #if self.total_epochs_processed % self.look_ahead_distance == 0:
        #    self.reset_dl_delay()
        return next_batch_id, next_batch_indices, isCached
    
    def _find_substitute_batch(self,next_batch_id, next_batch_indices):
        return next_batch_id, next_batch_indices, False
    
    