from collections import OrderedDict
from batch import Batch
import time
import threading
import logging
import datetime
from misc.priority_queue import PriorityQueue

class MLTrainingJob():
    def __init__(self,job_id,batch_size,look_ahead_distance,warm_up_distance):
        self.job_id = job_id
        self.batch_size = batch_size
        self.warm_up_distance = warm_up_distance
        self.look_ahead_distance=look_ahead_distance
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
        self.f_stop = True
        self.bacth_group_priorityq:PriorityQueue = None
        self.active = True
    
    def reset_epoch_timer(self):
        self.epoch_timer = time.time()

    def start_data_prep_service(self):
        self.f_stop = threading.Event()
        self.run_data_prep()
        
    def stop_data_prep_service(self):
        self.f_stop = True

    def run_data_prep(self):
        predicted_times ={}
        for idx in range(0,min(len(self.epoch_batches_remaining),self.look_ahead_distance)):
            batch_id = self.epoch_batches_remaining[idx]
            predicted_access_time = max(0,((idx * self.avg_training_speed) + self.data_laoding_delay) - (time.time() - self.epoch_timer))
            self.bacth_group_priorityq.push(batch_id, predicted_access_time)
            predicted_times[batch_id] = predicted_access_time
        
        logging.info(predicted_times)
        if not self.f_stop == True:
            # call f() again in 2 seconds
            threading.Timer(4, self.run_data_prep, []).start()

    def increment_epochs_processed(self):
        self.total_epochs_processed +=1
        if self.total_epochs_processed ==0:
            self.job_started_timestamp = datetime.datetime.now()

    def increment_batches_processed(self):
        self.total_batches_processed +=1

        if self.total_batches_processed == self.warm_up_distance:
            self.start_data_prep_service() 

    def set_batches_to_process(self,group_id, batches:dict[str,Batch],bacth_group_priorityq:PriorityQueue):
        self.current_epoch_batches = batches
        self.epoch_batches_remaining=list(batches.keys())
        self.current_batch_group = group_id
        self.bacth_group_priorityq = bacth_group_priorityq

    def set_avg_training_speed(self,speed):
        self.avg_training_speed = speed
    
    def update_data_laoding_delay(self,delay):
        self.data_laoding_delay += delay
    
    def reset_dl_delay(self):
        self.data_laoding_delay = 0
    
    def end_job(self):
        self.stop_data_prep_service()
        self.job_finished_timestamp = datetime.datetime.now()
        self.active = False
        return self.job_started_timestamp, self.job_finished_timestamp

    def _next_batch(self,use_substitutional_hits):
        
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
        
        return next_batch_id, next_batch_indices, isCached
    
    def _find_substitute_batch(self,next_batch_id, next_batch_indices):
        return next_batch_id, next_batch_indices, False
    
    