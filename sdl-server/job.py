
class MLTrainingJob():
    def __init__(self,job_id,batch_size):
        self.job_id = batch_size
        self.batch_size = batch_size
        self.training_speed = None
        self.total_epochs_processed = None
        self.current_group = None
        self.job_started_timestamp = None
        self.job_finished_timestamp = None
        #self.worker
    