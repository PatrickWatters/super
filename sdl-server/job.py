from collections import OrderedDict

class MLTrainingJob():
    def __init__(self,job_id,batch_size):
        self.job_id = job_id
        self.batch_size = batch_size
        self.avg_training_speed = 0
        self.data_laoding_delay = 0
        self.total_epochs_processed = -1
        self.total_batches_processed = -1
        self.current_batch_group = None
        self.job_started_timestamp = None
        self.job_finished_timestamp = None
        self.epoch_batches_remaining = OrderedDict()
        self.batch_groups_processed = []
    
    def assign_batches_for_epoch(self,epoch_batches):
        self.epoch_batches_remaining = epoch_batches
    
    def predict_batch_access_time(self,batch_idx):
        prediction = (batch_idx * self.avg_training_speed) + self.data_laoding_delay
        return prediction
    def reset_delay(self):
        self.data_laoding_delay = 0
        #is there a way I can easily get the index of the batch... ordered dictionary?
        #self.worker
    
    