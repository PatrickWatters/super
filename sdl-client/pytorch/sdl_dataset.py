import json
import torch
import functools
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
from PIL import Image
#from misc.time_helper import stopwatch
from torch.utils.data import Dataset
import io
from client import CMSClient
import base64
import time

class SDLDataset(Dataset):
    def __init__(self,job_id:int,blob_classes:dict, transform:Optional[Callable] =None,target_transform:Optional[Callable]=None):
        self.transform = transform
        self.target_transform = target_transform
        self._blob_classes = blob_classes
        self.job_id = job_id
        self.length = sum(len(class_items) for class_items in self._blob_classes.values())
    
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
        batch_data = input[3]

        if isCached:
            cache_hit = True
        else:
            cache_hit = False
        end = time.time()
        torch_imgs, torch_lables = self.convert_json_batch_to_torch_format(batch_data)
        prep_time = time.time() - end
        return torch_imgs, torch_lables, batch_id, cache_hit, prep_time

    def convert_json_batch_to_torch_format(self,batch_data):
        samples = json.loads(batch_data)
        imgs = []
        labels  =[]
        
        for img,label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)

            imgs.append(img)
            labels.append(label)

        return torch.stack(imgs), torch.tensor(labels)

'''
def __getitem__(self, input):
        batch_id = input[0]
        batch_indices = input[1]
        isCached = input[2]
        data = input[3]

        batch_data = redis_client.get_data(batch_id)
        if batch_data != None:
            cache_hit = True
            buffer = io.BytesIO(batch_data)
            batch_with_labels_dict = torch.load(buffer)
            return batch_with_labels_dict['inputs'],batch_with_labels_dict['labels'], batch_id,cache_hit
        else:
            return None, None, batch_id, isCached
'''