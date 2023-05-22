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

class CacheManagementService(pb2_grpc.CacheManagementServiceServicer):

    def __init__(self, *args, **kwargs):
        print(f'Running Setup for Cache Management Service')
        self._run_setup()
        self.training_jobs = {}
        self.drop_last = False
        pass

    def _run_setup(self):
        #check configuration
        self.config = self._read_config()
        if self.config is None:
            print(f'Unable to read config')
            return
        #check S3 bucket
        if not aws.s3bucket_exists(self.bucket_name):
            print(f'The bucket {self.bucket_name} does not exist or you have no access.')
            return
        #load training dataset S3 bucket
        self.train_dataset = Dataset(self.bucket_name,'train')
        print(f'Successfully loaded dataset from "{self.bucket_name}/train". Total files:{self.train_dataset.length}')

        #create lambda function if it doesn't already exisit

    def _read_config(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_filepath = dir_path+"/config.ini"
        # check if the config file exists
        if os.path.exists(config_filepath):
            config = ConfigParser()
            config.read(config_filepath)
            self.micro_batch_size = config["SUPER"]["micro_batch_size"]
            self.bucket_name = config["S3"]["bucket"]
            return config
        else:
             return None
    
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


    def GetServerResponse(self, request, context):
        # get the string from the incoming request
        message = request.message
        result = f'Hello I am up and running received "{message}" message from you'
        result = {'message': result, 'received': True}

        return pb2.MessageResponse(**result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CacheManagementServiceServicer_to_server(CacheManagementService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()



if __name__ == '__main__':
    serve()