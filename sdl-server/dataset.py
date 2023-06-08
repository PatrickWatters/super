import functools
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
import json
import boto3
from torch.utils.data.sampler import RandomSampler,BatchSampler, SequentialSampler
from batch import Batch, BatchGroup
from lambda_wrapper import LambdaWrapper
s3_client = boto3.client('s3')
s3Resource = boto3.resource("s3")

class Dataset():
    def __init__(self,s3_bucket_name,prefix, batch_size, drop_last, redis_port, redis_host, lambda_func_name='lambda_dataloader'):
        self.bucket_name = s3_bucket_name
        self.batch_size = batch_size
        self.drop_last =drop_last
        self.prefix = prefix
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        self._blob_classes = self._classify_blobs()
        self.length = sum(len(class_items) for class_items in self._blob_classes.values())
        self.redis_port =redis_port
        self.redis_host =redis_host
        self.lambda_func_name = lambda_func_name
        self.lambda_wrapper = LambdaWrapper()
    

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self._blob_classes.values())
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._blob_classes)
            for blob in self._blob_classes[blob_class]
        ]
    
    def _classify_blobs(self) -> Dict[str, List[str]]:

        blob_classes: Dict[str, List[str]] = {}
        #check if index file in the root of the folder to avoid having to loop through the entire bucket
        key = self.prefix + '/'+self.prefix+ '_index.json'
        content_object = s3Resource.Object(self.bucket_name, key)
        try:
            file_content = content_object.get()['Body'].read().decode('utf-8')
            blob_classes = json.loads(file_content) 
        except:
            print('No index in file S3 - loading dataset paths the long way')
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)        
            for page in pages:
                for blob in page['Contents']:
                    blob_path = blob.get('Key')
                    #check if the object is a folder which we want to ignore
                    if blob_path[-1] == "/":
                        continue
                    stripped_path = self._remove_prefix(blob_path, self.prefix).lstrip("/")
                    #Indicates that it did not match the starting prefix
                    if stripped_path == blob_path:
                        continue
                    if not self.is_image_file(blob_path):
                        continue
                    blob_class = stripped_path.split("/")[0]
                    blobs_with_class = blob_classes.get(blob_class, [])
                    blobs_with_class.append(blob_path)
                    blob_classes[blob_class] = blobs_with_class
        
        #json_object = json.dumps(blob_classes, indent=4)
        #with open("train_index.json", "w") as outfile:
        #    outfile.write(json_object)
        return blob_classes
    
    def _remove_prefix(self,s: str, prefix: str) -> str:
        if not s.startswith(prefix):
            return s
        return s[len(prefix) :]
    
    def generate_batches(self,seed:int, use_random_sampling:bool):
        setOfbatches={}
        if use_random_sampling:
            base_sampler = RandomSampler(self)
        else:
            base_sampler = SequentialSampler(self)
        
        batch_sampler = BatchSampler(base_sampler, batch_size=self.batch_size, drop_last=False)
        for i,batch_indiceis in enumerate(batch_sampler):
                batch_id = abs(hash(frozenset(batch_indiceis)))
                setOfbatches[batch_id] = Batch(id=batch_id,group_id=seed, indices=batch_indiceis)
        return setOfbatches
    
    def fetch_bacth_data(self, batch_id, batch_metadata,cache_after_retrevial=False):
            
        fun_params = {}
        fun_params['batch_metadata'] = batch_metadata
        fun_params['batch_id'] = batch_id
        fun_params['cache_bacth'] = cache_after_retrevial
        fun_params['return_batch_data'] = False
        fun_params['bucket'] = self.bucket_name
        fun_params['redis_host'] = self.redis_host 
        fun_params['redis_port'] = self.redis_port

        response = self.lambda_wrapper.invoke_function(self.lambda_func_name,fun_params)

        if 'errorMessage' in response:
            print(response['errorMessage'])
        else:
           return response['batch_data']    