import grpc
from concurrent import futures
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
from batch import BatchGroup
from misc.unique_priority_queue import UniquePriorityQueue
import time
import threading
from batch import Batch
from lambda_wrapper import LambdaWrapper

logging.basicConfig(format='%(asctime)s - %(message)s',filename='server.log', encoding='utf-8', level=logging.INFO)

class CacheManagementService(pb2_grpc.CacheManagementServiceServicer):

    def __init__(self, *args, **kwargs):
        print(f'Running Setup for Cache Management Service')
        self.batch_groups:Dict[str, BatchGroup] = {}
        self.active_training_jobs = {}
        self.global_batch_group_idx = 0
        self.global_queue = UniquePriorityQueue()
        self._read_config()
        self._check_environment()
        self._gen_new_group_of_batches() #create an initial set of batches
        #self.start_consuming()
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
            self.use_substitutional_hits = config.getboolean('SUPER','use_substitutional_hits')
            self.drop_last =  config.getboolean('SUPER','drop_last')
            #self.access_time_update_freq=float(config["SUPER"]["access_time_update_freq"])
            self.random_sampling_enabled=config.getboolean('SUPER','use_random_sampling')
            self.look_ahead_distance =int(config["SUPER"]["look_ahead_distance"])
            self.warm_up_distance=int(config["SUPER"]["warm_up_distance"])
            self.redis_port=int(config["SION"]["port"])
            self.redis_host=(config["SION"]["host"])
            self.lambda_func_name=config["SUPER"]["lambda_func_name"]

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
        self.train_dataset = Dataset(self.bucket_name,'train',self.micro_batch_size, drop_last=self.drop_last, redis_port=self.redis_port,
                                     redis_host= self.redis_host, lambda_func_name= self.lambda_func_name)
        print(f'Successfully loaded dataset from "{self.bucket_name}/train". Total files:{self.train_dataset.length}')

    def _gen_new_group_of_batches(self):
        self.global_batch_group_idx +=1 #TODO:
        new_batches = self.train_dataset.generate_batches(self.global_batch_group_idx, use_random_sampling = self.random_sampling_enabled)
        self.batch_groups[self.global_batch_group_idx] = BatchGroup(self.global_batch_group_idx,new_batches)

    def RegisterNewTrainingJob(self, request, context):
        job_id = request.job_id
        batch_size = request.batch_size 
        newjob = MLTrainingJob(job_id,batch_size,self.look_ahead_distance,self.warm_up_distance,global_priority_queue=self.global_queue)

        if self.drop_last:
            batches_per_epoch = len(self.train_dataset) // batch_size  # type: ignore[arg-type]
        else:
            batches_per_epoch = (len(self.train_dataset) + batch_size - 1) // batch_size  # type: ignore[arg-type]

        if job_id not in self.active_training_jobs:
            self.active_training_jobs[job_id] = newjob
            result = f'Job with Id {job_id} has been successfully regsistered!'
            result = {'message': result, 'dataset':json.dumps(self.train_dataset._blob_classes), 'registered': True,'batches_per_epoch': batches_per_epoch}    
        else:
            result = f'Job with Id {job_id} already exists!'
            result = {'message': result, 'dataset':'[]','registered': False,'batches_per_epoch': batches_per_epoch}
        return pb2.RegisterTrainingJobResponse(**result)
    
    def GetNextBatchForJob(self,request, context):
        training_job:MLTrainingJob = self.active_training_jobs[request.job_id]
        training_job.set_avg_training_speed(request.avg_training_speed)
        training_job.update_data_laoding_delay(request.prev_batch_dl_delay)
        training_job.increment_batches_processed()

        if len(training_job.currEpoch_remainingBatches) < 1: #batches exhausted, starting a new epoch
            self._assign_new_batches_to_job_for_epoch(training_job=training_job)
            training_job.reset_epoch_timer()
            training_job.reset_dl_delay()

        next_batch_id, next_batch_indices, isCached = training_job._next_batch(self.use_substitutional_hits)
        batch_data = None
        if not isCached:
            batch_data = self.train_dataset.fetch_bacth_data(next_batch_id,next_batch_indices,False)
        
        result = {'batch_id': str(next_batch_id),'batch_metadata': json.dumps(next_batch_indices), 'isCached': isCached,'batch_data': batch_data}
        logging.info({'batch_id': str(next_batch_id),'batch_metadata': json.dumps(next_batch_indices), 'isCached': isCached})

        return pb2.GetNextBatchForJobResponse(**result)
    
    def _assign_new_batches_to_job_for_epoch(self,training_job:MLTrainingJob):
        if training_job.currEpoch_batchGroup is not None:
            self.batch_groups[training_job.currEpoch_batchGroup].processed_by.append(training_job.job_id)

        if  training_job.job_id in self.batch_groups[self.global_batch_group_idx].processed_by:
            self._gen_new_group_of_batches()
        
        training_job.set_batches_for_new_epoch(self.global_batch_group_idx,self.batch_groups[self.global_batch_group_idx].batches)
        training_job.increment_epochs_processed()  

    
    def ProcessJobEndedMessage(self, request, context):
        job_id = request.job_id
        job:MLTrainingJob = self.active_training_jobs[job_id]
        start_time, finish_time, duration = job.end_job()
        del self.active_training_jobs[job_id]

        result = f'Job with Id {job_id} finished! start time: {start_time}, finish time: {finish_time}, duration: {duration}'
        result = {'message': result, 'received': True}
        return pb2.MessageResponse(**result)

    def GetServerResponse(self, request, context):
        # get the string from the incoming request
        message = request.message
        result = f'Hello I am up and running received "{message}" message from you'
        result = {'message': result, 'received': True}
        return pb2.MessageResponse(**result)
    
    def start_consuming(self):
        consumers = []
        for i in range(1):
            name = 'Consumer-{}'.format(i)
            c = ConsumerThread(name=name, bucket_name=self.bucket_name, redis_host=self.redis_host, redis_port=self.redis_port, queue=self.global_queue, batch_groups=self.batch_groups)
            c.start()
            consumers.append(c)

        for consumer in consumers:
            consumer.join()
    
class ConsumerThread(threading.Thread):
    def __init__(self, name, bucket_name,redis_host,redis_port,queue,batch_groups, group=None, target=None, args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.q:UniquePriorityQueue = queue
        self.batch_groups=batch_groups
        self.target = target
        self.name = name
        self.lambda_wrapper = LambdaWrapper()
        self.bucket_name = bucket_name
        self.redis_host = redis_host
        self.redis_port = redis_port
        return
    
    def run(self):
        while True:
            item = self.q.get()
            #decide what to do next..
            item = list(item)
            priority, task = item
            job_id, group_id, batch_id = task
            batch_in_question:Batch = self.batch_groups[group_id][batch_id]
                
            if not batch_in_question.isCached:
                print(self.name + ' getting ' + str(item)  + ' : ' + str(self.q.qsize()) + ' items in queue')

                fun_params = {}
                fun_params['batch_metadata'] = batch_in_question.labelled_paths
                fun_params['batch_id'] = batch_in_question.batch_id
                fun_params['cache_bacth'] = True
                fun_params['return_batch_data'] = False
                fun_params['bucket'] = self.bucket_name
                fun_params['redis_host'] = self.redis_host 
                fun_params['redis_port'] = self.redis_port
                self.lambda_wrapper.invoke_function()
                batch_in_question.isCached = True


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CacheManagementServiceServicer_to_server(CacheManagementService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()