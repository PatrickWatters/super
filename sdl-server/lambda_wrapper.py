from botocore.exceptions import ClientError
import logging
import boto3
import json
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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