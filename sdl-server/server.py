import grpc
from concurrent import futures
import time
import cms_pb2_grpc as pb2_grpc
import cms_pb2 as pb2
from configparser import ConfigParser
import os.path
import misc.aws as aws
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
from dataset import Dataset
from job import MLTrainingJob
import logging
import json
from batch_group import BatchGroup
import threading
logging.basicConfig(format='%(asctime)s - %(message)s',filename='server.log', encoding='utf-8', level=logging.INFO)

class CacheManagementService(pb2_grpc.CacheManagementServiceServicer):

    def __init__(self, *args, **kwargs):
        print(f'Running Setup for Cache Management Service')
        self.batch_groups = {}
        self.training_jobs = {}
        self.drop_last = None
        self.global_batch_group_idx = 0
        self.use_substitutional_hits = None
        self.micro_batch_size = None
        self.bucket_name = None
        self.access_time_update_freq = None
        self._read_config()
        self._check_environment()
        self._create_group_of_batches() #create an initial set of batches
        pass

    def _read_config(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_filepath = dir_path+"/config.ini"
        # check if the config file exists
        if os.path.exists(config_filepath):
            config = ConfigParser()
            config.read(config_filepath)
            self.micro_batch_size = int(config["SUPER"]["micro_batch_size"])
            self.bucket_name = config["S3"]["bucket"]
            self.use_substitutional_hits = bool(config["SUPER"]["use_substitutional_hits"])
            self.drop_last = bool(config["SUPER"]["drop_last"])
            self.access_time_update_freq=int(config["SUPER"]["access_time_update_freq"])
            return config
        else:
            print(f'Unable to read config')
            return

    def _check_environment(self):
        #check S3 bucket
        if not aws.s3bucket_exists(self.bucket_name):
            print(f'The bucket {self.bucket_name} does not exist or you have no access.')
            return
        #load training dataset S3 bucket
        self.train_dataset = Dataset(self.bucket_name,'train',self.micro_batch_size, drop_last=self.drop_last)
        print(f'Successfully loaded dataset from "{self.bucket_name}/train". Total files:{self.train_dataset.length}')

    def _create_group_of_batches(self):
        self.global_batch_group_idx +=1 #TODO:
        new_batches = self.train_dataset.create_group_of_batches(self.global_batch_group_idx)
        self.batch_groups[self.global_batch_group_idx] = BatchGroup(self.global_batch_group_idx,new_batches)

    
    def RegisterNewTrainingJob(self, request, context):
        job_id = request.job_id
        batch_size = request.batch_size 
        newjob = MLTrainingJob(job_id,batch_size)

        if self.drop_last:
            batches_per_epoch = len(self.train_dataset) // batch_size  # type: ignore[arg-type]
        else:
            batches_per_epoch = (len(self.train_dataset) + batch_size - 1) // batch_size  # type: ignore[arg-type]

        if job_id not in self.training_jobs:
            self.training_jobs[job_id] = newjob
            result = f'Job with Id {job_id} has been successfully regsistered!'
            result = {'message': result, 'dataset':json.dumps(self.train_dataset._blob_classes), 'registered': True,'batches_per_epoch': batches_per_epoch}    
        else:
            result = f'Job with Id {job_id} already exists!'
            result = {'message': result, 'dataset':'[]','registered': False,'batches_per_epoch': batches_per_epoch}
        return pb2.RegisterTrainingJobResponse(**result)
    
    def GetNextBatchForJob(self,request, context):
        training_job:MLTrainingJob = self.training_jobs[request.job_id]
        training_job.avg_training_speed = request.avg_training_speed
        training_job.data_laoding_delay += request.prev_batch_dl_delay
        training_job.total_batches_processed +=1
        
        if len(training_job.epoch_batches_remaining.items()) < 1: #batches exhausted, job starting a new epoch
            self._set_batches_for_new_epoch(training_job=training_job)
            training_job.reset_delay()
        
        batch_group:BatchGroup = self.batch_groups[training_job.current_batch_group]
        next_batch_id = next(iter(training_job.epoch_batches_remaining))
        isCached = batch_group.batchIsCached(next_batch_id)

        if isCached == False and self.use_substitutional_hits:
            next_batch_id,isCached = self._find_substitute_batch(batch_group, training_job.epoch_batches_remaining, next_batch_id)
        
        if  training_job.total_batches_processed > 0 and training_job.total_batches_processed % self.access_time_update_freq == 0:
            self._estimate_batch_access_times(request.job_id)
        #self.start_thread(self._estimate_batch_access_times,name=None,args=[request.job_id])
        
        
        result = {'batch_id': str(next_batch_id),'batch_metadata': json.dumps(training_job.epoch_batches_remaining.pop(next_batch_id)), 'isCached': isCached}
        logging.info(result)
        return pb2.GetNextBatchForJobResponse(**result)
    
    def _find_substitute_batch(self,batch_group:BatchGroup,epoch_batches_remaining, orgional_batch_id):
        next_batch_id = orgional_batch_id
        isCached = False
        for batch_id in epoch_batches_remaining:
            if batch_group.batchIsCached(batch_id):
                next_batch_id = batch_id
                isCached = True
                break
        return next_batch_id,isCached
    
    def _set_batches_for_new_epoch(self,training_job:MLTrainingJob):
        if training_job.current_batch_group is not None:
            training_job.batch_groups_processed.append(training_job.current_batch_group)
            
        if self.global_batch_group_idx in training_job.batch_groups_processed:
            self._create_group_of_batches()

        training_job.epoch_batches_remaining = self.batch_groups[self.global_batch_group_idx].batches_dict
        training_job.current_batch_group = self.global_batch_group_idx        
        training_job.total_epochs_processed +=1

    def GetServerResponse(self, request, context):
        # get the string from the incoming request
        message = request.message
        result = f'Hello I am up and running received "{message}" message from you'
        result = {'message': result, 'received': True}
        return pb2.MessageResponse(**result)
    
    def _estimate_batch_access_times(self,job_id):
        job:MLTrainingJob = self.training_jobs[job_id]
        batchGroup:BatchGroup = self.batch_groups[job.current_batch_group]
        indx= 0
        for bacth_id in job.epoch_batches_remaining:
            indx+=1
            prediction = job.predict_batch_access_time(batch_idx=indx)
            print(prediction)
            batchGroup.update_batch_access_estimate(bacth_id,prediction)
        job.reset_delay()

    def start_thread(self, func, name=None, args = []):
        threading.Thread(target=func, name=name, args=args).start()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CacheManagementServiceServicer_to_server(CacheManagementService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()



if __name__ == '__main__':
    serve()