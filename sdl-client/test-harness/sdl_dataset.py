import json
import torch
import functools
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
from PIL import Image
#from misc.time_helper import stopwatch
from torch.utils.data import Dataset
from redis_client import RedisClient
import io
from client import CMSClient
redis_client = RedisClient('local')

class SDLDataset(Dataset):
    def __init__(self,job_id:int,blob_classes:dict,client:CMSClient, transform:Optional[Callable] =None,target_transform:Optional[Callable]=None):
        self.transform = transform
        self.target_transform = target_transform
        self._blob_classes = blob_classes
        self.job_id = job_id
        self.length = sum(len(class_items) for class_items in self._blob_classes.values())
        self.sdl_cleint = client
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._blob_classes)
            for blob in self._blob_classes[blob_class]
        ]
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self._blob_classes.values())
    
    def __getitem__(self, input):
        batch_id = input[0]
        batch_indices = input[1]
        isCached = input[2]
        batch_data = redis_client.get_data(batch_id)
        if batch_data != None:
            cache_hit = True
            buffer = io.BytesIO(batch_data)
            batch_with_labels_dict = torch.load(buffer)
            return batch_with_labels_dict['inputs'],batch_with_labels_dict['labels'], batch_id,cache_hit
        else:
            return None, None, batch_id, isCached
