from botocore.exceptions import ClientError
import logging
import boto3
import json
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from PIL import Image
import base64
import io
import pathlib
import time

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
  


class LambdaWrapper:
    def __init__(self,bucket_name,redis_host,redis_port, lambda_client=boto3.client('lambda'), iam_resource=boto3.resource('iam'),
                 function_name='lambda_dataloader'):
        self.lambda_client = lambda_client
        self.iam_resource = iam_resource
        self.function_name = function_name
        self.bucket_name = bucket_name
        self.redis_host = redis_host
        self.redis_port = redis_port

    def invoke_function(self, labelled_paths, batch_id, cache_after_retrevial, include_batch_data_in_response = True, get_log=False):
        fun_params = {}
        fun_params['batch_metadata'] = labelled_paths
        fun_params['batch_id'] = batch_id
        fun_params['cache_bacth'] = cache_after_retrevial
        fun_params['return_batch_data'] = include_batch_data_in_response
        fun_params['bucket'] = self.bucket_name
        fun_params['redis_host'] = self.redis_host 
        fun_params['redis_port'] = self.redis_port
   
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                Payload=json.dumps(fun_params),
                LogType='Tail' if get_log else 'None')
            #logger.info("Invoked function %s.", function_name)
        except ClientError:
            logger.exception("Couldn't invoke function %s.", self.function_name)
            raise
        return response
    
    def fetch_from_local_disk(self, labelled_paths, batch_id, cache_after_retrevial, include_batch_data_in_response = True, get_log=False,redis_client=None):
        time.sleep(2) #simulation the time it would take to load from S3
        fun_params = {}
        fun_params['batch_metadata'] = labelled_paths
        fun_params['batch_id'] = batch_id
        fun_params['cache_bacth'] = cache_after_retrevial
        fun_params['return_batch_data'] = include_batch_data_in_response
        fun_params['bucket'] = self.bucket_name
        fun_params['redis_host'] = self.redis_host 
        fun_params['redis_port'] = self.redis_port
        response = dict()
        samples = []
        cache_error_message = ''
        batch_metadata = fun_params['batch_metadata']
        batch_id = fun_params['batch_id']
        cache_bacth= fun_params['cache_bacth'] 
        return_batch_data= fun_params['return_batch_data'] 
        bucket_name=fun_params['bucket']

        for path, label in batch_metadata:        
            path = "/Users/patrickwatters/Projects/datasets/{}/{}".format(bucket_name, path)
            file_extension = pathlib.Path(path).suffix.replace('.','')
            
            img = Image.open(path)
            if img.mode == "L":
                    img = img.convert("RGB")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=file_extension)
            base_64_encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            samples.append((base_64_encoded_img, label))
        
        json_samples = json.dumps(samples) 
        isCached = False

        if cache_bacth and redis_client.isLocal:      
            try:
                redis_client.set_data(batch_id, json_samples)
                isCached = True
            except Exception as e:
                isCached = False
                cache_error_message = str(e)  

        if not isCached or return_batch_data:
                response_msg = {'batch_id': batch_id,'isCached': isCached,'batch_data':json_samples,'cache_error_message':cache_error_message}
        else:
                response_msg = {'batch_id': batch_id,'isCached': isCached,'batch_data':json_samples,'cache_error_message':cache_error_message}

                #response_msg = {'batch_id': batch_id,'isCached': isCached,'cache_error_message':cache_error_message}
        
        response['Payload'] = json.dumps(response_msg)
        return response