from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
import json
from batch import BatchSet, Batch
from torch.utils.data.sampler import RandomSampler,BatchSampler, SequentialSampler
import time
import logging
import sys
import functools
from pathlib import Path
import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class DataFeedCoordinator():
    def __init__(self,args):
        self.global_epoch_counter = 0
        self.batch_size = args.batch_size
        self.drop_last =False
        self.prefix = 'train'
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        if args.source_system == 's3':
            self.s3_bucket = args.s3_bucket
            self._blob_classes = self._classify_blobs_s3()
        else:
            self.data_dir = args.data_dir + '/' + self.prefix
            self._blob_classes = self._classify_blobs_local()
        self.use_random_sampling = False
        self.batchSets:Dict[str, BatchSet] = {}
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self._blob_classes.values())
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._blob_classes)
            for blob in self._blob_classes[blob_class]
        ]
    
    def _remove_prefix(self,s: str, prefix: str) -> str:
        if not s.startswith(prefix):
            return s
        return s[len(prefix) :]
    
    def _classify_blobs_local(self) -> Dict[str, List[str]]:
        blob_classes: Dict[str, List[str]] = {}
        index_file = Path(self.data_dir + '/train_index.json')
        if(index_file.exists()):
          f = open(index_file.absolute())
          blob_classes = json.load(f)
        else:
            for filename in glob.iglob(self.data_dir + '**/**', recursive=True):
                #check if the object is a folder which we want to ignore
                if filename[-1] == "/":
                    continue
                if not self.is_image_file(filename):
                    continue
            
                stripped_path = self._remove_prefix(filename, self.data_dir).lstrip("/")
                blob_class = stripped_path.split("/")[0]
                blobs_with_class = blob_classes.get(blob_class, [])
                blobs_with_class.append(filename)
                blob_classes[blob_class] = blobs_with_class
        return blob_classes
    
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def _classify_blobs_s3(self) -> Dict[str, List[str]]:
         
        logger.info("Reading dataset from S3 bucket:{}.".format(self.s3_bucket))
        end = time.time()

        import boto3
        s3_client = boto3.client('s3')
        s3Resource = boto3.resource("s3")
        blob_classes: Dict[str, List[str]] = {}
        #check if index file in the root of the folder to avoid having to loop through the entire bucket
        key = self.prefix + '/'+self.prefix+ '_index.json'
        content_object = s3Resource.Object(self.s3_bucket, key)
        try:
            file_content = content_object.get()['Body'].read().decode('utf-8')
            blob_classes = json.loads(file_content) 
        except:
            logger.debug('No index in file S3 - loading dataset paths the long way')
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.prefix)        
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
        self.length = sum(len(class_items) for class_items in blob_classes.values())

        logger.info("Finished reading dataset,Total Files:{},Elasped Time:{}".format(self.length,round(time.time()-end,2)))

        #json_object = json.dumps(blob_classes, indent=4)
        #with open("train_index.json", "w") as outfile:
        #    outfile.write(json_object)
        return blob_classes
    

    def gen_new_bacthes_for_job(self, jobId, jobBatchSetId):

        if jobBatchSetId is not None:  
            self.batchSets[jobBatchSetId].finshedProcessing.append(jobId)
        
        if self.global_epoch_counter == 0 or jobId in self.batchSets[self.global_epoch_counter].finshedProcessing:
            self.global_epoch_counter +=1
            newBatchSet = BatchSet(self.global_epoch_counter)

            if self.use_random_sampling:
                base_sampler = RandomSampler(self)
            else:
                base_sampler = SequentialSampler(self)
            
            batch_sampler = BatchSampler(base_sampler, batch_size=self.batch_size, drop_last=False)
        
            for i,batchIndiceis in enumerate(batch_sampler):
                id = abs(hash(frozenset(batchIndiceis)))
                labelled_paths =[]
                for id in batchIndiceis:
                    labelled_paths.append(self._classed_items[id])
                newBatch = Batch(batchId=id,setId=newBatchSet.setId,indices=batchIndiceis, labelled_paths=labelled_paths)
                newBatchSet.batches[id] = newBatch
            self.batchSets[newBatchSet.setId] = newBatchSet
            return newBatchSet.batches.keys(),newBatchSet.setId
        else:
            return self.batchSets[self.global_epoch_counter].batches.keys(),self.global_epoch_counter
    
    def batch_is_cached(self, bacthSetId, batchId):
        #check if batch is cached or is in the process of being cached
        return self.batchSets[bacthSetId].batches[batchId].isCached
    
    def batch_is_inProgress(self, bacthSetId, batchId):
        #check if batch is cached or is in the process of being cached
        return self.batchSets[bacthSetId].batches[batchId].isCached
    
    def set_batch_inProgress(self, bacthSetId, batchId, status):
        self.batchSets[bacthSetId].batches[batchId].setInProgessStatus(isInProgress=status)
    
    def set_batch_isCached(self, bacthSetId, batchId, status):
        self.batchSets[bacthSetId].batches[batchId].setCachedStatus(isCached=status)
    
    def update_batch_last_access_time(self, bacthSetId, batchId):
        self.batchSets[bacthSetId].batches[batchId].updateLastPinged()
    
    def get_batch_lablled_paths(self, bacthSetId, batchId):
        return self.batchSets[bacthSetId].batches[batchId].labelled_paths