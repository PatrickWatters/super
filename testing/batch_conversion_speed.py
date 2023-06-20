import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import io
import base64
import time
import redis
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
import threading


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
                print(str(e))
                return False
        
    def get_data(self, key):
        try:
            return self.conn.get(key)
    
        except Exception as e:

                print(str(e))
                return False
        





def process_batch(num_times,torch_format=False, transforms=None, multithreaded = False, num_workers = False):
    results={}

    redis_c = RedisClient(redis_host='127.0.0.1',redis_port=6379)
    if torch_format:
         label = 'torch format'
         key = 4384029838768895551
    else:
         label = 'pil format'
         key = 'base64'

    batch_data = redis_c.get_data(key)
    end = time.time()
    
    for idx in range(0, num_times):
        if torch_format:
              round_end = time.time()
              decoded_data = base64.b64decode(batch_data)
              buffer = io.BytesIO(decoded_data)
              decoded_batch = torch.load(buffer)
              batch_imgs = decoded_batch['inputs']
              batch_labels = decoded_batch['labels']
              buffer.close()
              results[idx] = time.time() - round_end
              print("Torch-Round{}:{}".format(idx,time.time() - round_end))
        else:
              round_end = time.time()
              torch_imgs, torch_lables = convert_json_batch_to_torch_format(batch_data, transform)
              results[idx] = time.time() - round_end
              print("Base64-Round {}:{}".format(idx,time.time() - time.time() - round_end))

    print("{}: Total time for {} rounds:{}".format(label,num_times,time.time() - end))
    return results


def convert_json_batch_to_torch_format(batch_data, transform):
        samples = json.loads(batch_data)
        imgs = []
        labels  =[]
        transform = None
        for img,label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            if transform is not None:
                img = transform(img)
            imgs.append(img)
            labels.append(label)
        #return torch.stack(imgs), torch.tensor(labels)
        return imgs, labels

if __name__ == "__main__":
     
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [#transforms.Resize(256), 
         #transforms.CenterCrop(224), 
         transforms.ToTensor(), normalize]
    )

    torch_times = process_batch(num_times=1000, torch_format=True, transforms=None)
    pil_times =   process_batch(num_times=1000, torch_format=False, transforms=transform)

    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(list(torch_times.keys()),list(torch_times.values()), label='Tensor Format')
    plt.plot(list(pil_times.keys()),list(pil_times.values()),label='PIL Format')
    plt.xlabel('Batch Idx')
    plt.ylabel('Data Augmentation Time (s)')
    plt.title("Data Augmentation Performance")
    plt.legend()
    plt.show()
    print('')
