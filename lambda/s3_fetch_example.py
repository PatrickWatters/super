import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from PIL import Image
import logging
import os
import json
import base64
import sys
import gc
import boto3
from boto3.s3.transfer import TransferConfig
import torch
import torchvision.transforms as transforms
import numpy as np
# Set the desired multipart threshold value (5GB)
GB = 1024 ** 3
config = TransferConfig(use_threads=True,max_concurrency=10,multipart_threshold=5*GB) # To consume less downstream bandwidth, decrease the maximum concurrency

# ===================================== Setings ==================================================================
s3 = boto3.client('s3')
# ===================================== Setings ==================================================================
logger = logging.getLogger()

def example_lambda_handler(event, context):
    # Set the log level based on a variable configured in the Lambda environment.
    logger.setLevel(os.environ.get('LOG_LEVEL', logging.INFO))
    logger.debug('Event: %s', event)
    s3_bucket=event['bucket']
    batch_labelledItems = event['batch_labelledItems']
    add_to_cache = event['cache_bacth']
    return_batch_data= event['return_batch_data']
    laod_as_torch=event['laod_as_torch']

    if laod_as_torch:
        imgs =[]
        labels =[]
        for key, result in fetch_data_from_s3(s3_bucket,batch_labelledItems,load_img_as_tensor):
            imgs.append(result[0])
            labels.append(result[1])

        with io.BytesIO() as f:
            torch.save({'inputs': torch.stack(imgs), 'labels': torch.tensor(labels)}, f)
            base_64_encoded_batch = base64.b64encode(f.getvalue()).decode('utf-8')
        return base_64_encoded_batch   

    else:
        samples =[]
        for key, result in fetch_data_from_s3(s3_bucket,batch_labelledItems, load_img_as_base64):
            samples.append(result)
        json_samples = json.dumps(samples)
        return json_samples


def tensor_to_string(data):
    buff = io.BytesIO()
    torch.save(data, buff)
    tensor_as_string = base64.b64encode(buff.getvalue()).decode('utf-8')
    decode =  tensor_as_string.decode("utf-8")
    return decode

def fetch_data_from_s3(s3_bucket,batch_labelledItems,func_name):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_key = {executor.submit(func_name, key,s3_bucket): key for key in batch_labelledItems}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()
            if not exception:
                yield key, future.result()
            else:
                yield key, exception

def size_of_tensor_in_bytes(encoding):
    return (encoding.nelement() * encoding.element_size())/1048576

def load_img_as_base64(labelledItem, s3_bucket):
    file_path = labelledItem[0]
    label = labelledItem[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        base_64_encoded_img = base64.b64encode(f.getvalue()).decode('utf-8')
    return base_64_encoded_img, label

def load_img_as_numpy(labelledItem, s3_bucket):
    file_path = labelledItem[0]
    label = labelledItem[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        img = Image.open(io.BytesIO(f.getvalue()))
        pil_img = img.convert('RGB')
    return np.array(img)



def load_img_as_PIL(labelledItem, s3_bucket):
    file_path = labelledItem[0]
    label = labelledItem[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        img = Image.open(io.BytesIO(f.getvalue()))
        pil_img = img.convert('RGB')
    return pil_img, label

def load_img_as_tensor(labelledItem, s3_bucket):
    file_path = labelledItem[0]
    label = labelledItem[1]
    with io.BytesIO() as f:
        s3.download_fileobj(s3_bucket,file_path,f, Config=config)
        img = Image.open(io.BytesIO(f.getvalue()))
        pil_img = img.convert('RGB')
    
    transformation = transforms.Compose([transforms.ToTensor(),])
    tensor_img = transformation(pil_img)
    return tensor_img, label


if __name__ == "__main__":
    #create some dummy data
    torch_test = False
    batch_labelledItems = []
    for i in range(0,10):
        batch_labelledItems.append(('test/Frog/leopard_frog_s_001876.png',6))

    action_params = {}
    action_params['batch_labelledItems'] = batch_labelledItems
    action_params['cache_bacth'] = False
    action_params['return_batch_data'] = True
    action_params['bucket'] = 'sdl-cifar10'    
    action_params['laod_as_torch'] = torch_test

    if torch_test:
        #torch response type as base 64 encoded string
        lambda_response = example_lambda_handler(action_params, None)
        batch_data = base64.b64decode(lambda_response)
        buffer = io.BytesIO(batch_data)
        decoded_batch = torch.load(buffer)
        batch_imgs = decoded_batch['inputs']
        batch_labels = decoded_batch['labels']
        print(size_of_tensor_in_bytes(batch_imgs))

        print()
    else:
        samples = json.loads(example_lambda_handler(action_params, None))
        tensor_imgs = []
        labels  =[]
        convert_tensor = transforms.ToTensor()
        for img,label in samples:
            pil_img = Image.open(io.BytesIO(base64.b64decode(img)))
            tensor_imgs.append(convert_tensor(pil_img))
            labels.append(label)

        batch_imgs = torch.stack(tensor_imgs)
        batch_labels = torch.tensor(labels)
