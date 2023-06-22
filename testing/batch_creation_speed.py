import time
from PIL import Image
import pathlib
import base64
import json
import io
import functools
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
import json
import boto3
from torch.utils.data.sampler import RandomSampler,BatchSampler, SequentialSampler
from logging import error
import redis
import logging
import torch
import torchvision.transforms as transforms

class RedisClient:

    def __init__(self, redis_host, redis_port ):
        self.exire_time = 0
        self.conn = redis.StrictRedis(host=redis_host, port=redis_port)
        self.isLocal = redis_host == '127.0.0.1'
   

    def set_data(self, key, value):
        try:
            self.conn.set(key, value)
            return True
        except Exception as e:
                logging.error(str(e))
                print(str(e))
                return False
        
    def get_data(self, key):
        try:
            return self.conn.get(key)
    
        except Exception as e:
                logging.error(str(e))
                print(str(e))
                return False
        
s3_client = boto3.client('s3')
s3Resource = boto3.resource("s3")

class Dataset():
    def __init__(self,s3_bucket_name,prefix, batch_size, use_random_sampling:bool, drop_last):
        self.bucket_name = s3_bucket_name
        self.batch_size = batch_size
        self.drop_last =drop_last
        self.prefix = prefix
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        self._blob_classes = self._classify_blobs()
        self.length = sum(len(class_items) for class_items in self._blob_classes.values())
        self.use_random_sampling = use_random_sampling
    

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
    
    def generate_set_of_batches(self,seed:int):
        setOfbatches={}      
        if self.use_random_sampling:
            base_sampler = RandomSampler(self)
        else:
            base_sampler = SequentialSampler(self)
        
        batch_sampler = BatchSampler(base_sampler, batch_size=self.batch_size, drop_last=False)
        
        for i,batch_indiceis in enumerate(batch_sampler):
                batch_id = abs(hash(frozenset(batch_indiceis)))
                labelled_paths =[]
                for id in batch_indiceis:
                    labelled_paths.append(self._classed_items[id])
        return labelled_paths

def fetch_from_local_disk(num_times, labelled_paths, batch_id,bucket_name ): 
        
        tend = time.time()
        #simulation the time it would take to load from S3
        fun_params = {}
        fun_params['batch_metadata'] = labelled_paths
        fun_params['batch_id'] = batch_id
        fun_params['cache_bacth'] = False
        fun_params['return_batch_data'] = False
        fun_params['bucket'] = bucket_name

        samples = []
        batch_metadata = fun_params['batch_metadata']
        batch_id = fun_params['batch_id']
        bucket_name=fun_params['bucket']
        
        for i in range(0, num_times):
            end = time.time()
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
            print("Image loading:{}".format(time.time() - end))
        print("total time:{}".format(time.time() - tend))


def fetch_from_redis(um_times, batch_ids):
    end = time.time()
    total_fetch_fromredis = 0
    total_process = 0
    cli = RedisClient(redis_host='54.201.114.145', redis_port='6378')
    for id in batch_ids:
        fend =time.time()
        batch_data = cli.get_data(id)
        total_fetch_fromredis += time.time() - fend
        pend =time.time()
        torch_imgs, torch_lables = convert_json_batch_to_torch_format(batch_data)
        total_process += time.time() - pend
        print("Image loading: {}, PreProcessing: {}".format(total_fetch_fromredis,total_process))

    print("Image loading:{}".format(time.time() - end))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
        [#transforms.Resize(256), 
         #transforms.CenterCrop(224), 
         transforms.ToTensor(), normalize]
    )
def convert_json_batch_to_torch_format(batch_data):
        samples = json.loads(batch_data)
        imgs = []
        labels  =[]
        
        for img,label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            img = transform(img)

            imgs.append(img)
            labels.append(label)

        return torch.stack(imgs), torch.tensor(labels)

if __name__ == "__main__":
    with open('testing/batchids.txt') as file:
     lines = [line.rstrip() for line in file]
    fetch_from_redis(0, lines)

    f = open('testing/batchpaths.json')
    data = json.load(f)
    resposne = fetch_from_local_disk(196, data,'1', 'sdl-cifar10')
    assert()