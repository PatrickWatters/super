
from super_lambda_mgmt import SUPERLambdaMgmt
import random
import time
#import torch
import base64
import io
import json
#import torchvision.transforms as transforms
from PIL import Image
import logging
import csv

def size_of_str(input_string, metric ='bytes'):
    res = len(input_string.encode('utf-8'))
    return convert_bytes(res, metric)

def convert_bytes(byte_value, metric):
    if metric == 'kb':
         return byte_value/1024
    elif metric == 'mb':
         return byte_value/1048576
    elif metric == 'gb':
         return byte_value/1073741824
    else:
         return byte_value

def size_of_tensor_in_bytes(encoding, metric ='bytes'):
    size_in_bytes = encoding.nelement() * encoding.element_size()
    return convert_bytes(size_in_bytes, metric)
 

def process_response(batch_data:str):
    samples = json.loads(batch_data)
    tensor_imgs = []
    labels  =[]
    #convert_tensor = transforms.ToTensor()
    
    #for img,label in samples:
    #        pil_img = Image.open(io.BytesIO(base64.b64decode(img)))
    #        tensor_imgs.append(convert_tensor(pil_img))
    #        labels.append(label)

    #batch_imgs = torch.stack(tensor_imgs)
    #batch_labels = torch.tensor(labels)

    #print(size_of_tensor_in_bytes(batch_imgs,'mb'))
    #print(size_of_tensor_in_bytes(batch_labels,'mb'))



def gen_dummy_batch(batch_size=256):
    batch_metadata = []
    for i in range(0,batch_size):
        batch_metadata.append(('test/Frog/leopard_frog_s_001876.png',6))
    return batch_metadata

def simple_invokation(action_params,batch_size):
    mgr = SUPERLambdaMgmt()
    print(action_params['batch_id'])
    batch_metadata = gen_dummy_batch(batch_size)
    action_params['batch_metadata'] = batch_metadata

    #mgr.update_function()
    end = time.time()
    #call function
    response = mgr.invoke_function(action_params, False)
    if 'errorMessage' in response:
        print(response['errorMessage'])
        logging.info(response['errorMessage'])
    if response['isCached'] == False:
        print('Batch Size (bytes):', size_of_str(response['batch_data']))
        print('Response Size (MB):', size_of_str(response['batch_data'],'mb'))
    else:
        if action_params['return_batch_data']:
            print('Batch Size (bytes):', size_of_str(response['batch_data']))
            print('Response Size (MB):', size_of_str(response['batch_data'],'mb'))
            process_response(response['batch_data'])

    print(time.time() - end)

def populate_cache_until_full(action_params,batch_size,outputfile, stop_after = None):
    mgr = SUPERLambdaMgmt()
    batch_metadata = gen_dummy_batch(batch_size)
    action_params['batch_metadata'] = batch_metadata
    tend = time.time()
    counter =0
    total_size = 0
    result = []
    while (True):
        if stop_after is not None:
            if counter == stop_after:
                break
        action_params['batch_id'] = counter+1
        single_batch_time = time.time()
        response = mgr.invoke_function(action_params, False)
        if 'errorMessage' in response:
            print(response['errorMessage'])
            logging.info(response['errorMessage'])
            break
        if response['isCached'] == False:
            break
        else:
            requesttime =time.time()-single_batch_time
            if action_params['return_batch_data']:
                total_size+=size_of_str(response['batch_data'],'mb')
            else:
                total_size+= 0.794189453125
            counter +=1
            total_size +=0.03134
            result.append((action_params['batch_id'],total_size,requesttime,time.time()-tend))
            logging.info((counter,total_size,requesttime,time.time()-tend))

    with open(outputfile, 'w') as f:
        filewriter = csv.writer(f, delimiter='\t',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['BatchId','Total Cached (Mb)','Batch Load Time (s)', 'Elapsed Time (s)'])
        for val in result:
                filewriter.writerow([val[0],val[1],val[2],val[3]])


def populate_cache_multithreaded(action_params,batch_size,outputfile, stop_after = None):
    mgr = SUPERLambdaMgmt()
    batch_metadata = gen_dummy_batch(batch_size)
    action_params['batch_metadata'] = batch_metadata
    tend = time.time()
    counter =0
    total_size = 0
    result = []
    while (True):
        if stop_after is not None:
            if counter == stop_after:
                break
        action_params['batch_id'] = counter+1
        single_batch_time = time.time()
        response = mgr.invoke_function(action_params, False)
        if 'errorMessage' in response:
            print(response['errorMessage'])
            logging.info(response['errorMessage'])
            break
        if response['isCached'] == False:
            break
        else:
            requesttime =time.time()-single_batch_time
            if action_params['return_batch_data']:
                total_size+=size_of_str(response['batch_data'],'mb')
            else:
                total_size+= 0.794189453125
            counter +=1
            total_size +=0.03134
            result.append((action_params['batch_id'],total_size,requesttime,time.time()-tend))
            logging.info((counter,total_size,requesttime,time.time()-tend))

    with open(outputfile, 'w') as f:
        filewriter = csv.writer(f, delimiter='\t',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['BatchId','Total Cached (Mb)','Batch Load Time (s)', 'Elapsed Time (s)'])
        for val in result:
                filewriter.writerow([val[0],val[1],val[2],val[3]])




if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',filename='cache_load.log', encoding='utf-8', level=logging.INFO)
    batch_metadata = gen_dummy_batch(256)
    action_params = {}
    action_params['batch_metadata'] = batch_metadata
    action_params['batch_id'] = random.randint(0,10000000000)
    action_params['cache_bacth'] = True
    action_params['return_batch_data'] = True
    action_params['bucket'] = 'sdl-cifar10'    
    action_params['redis_host'] = 'super-redis.rdior4.ng.0001.usw2.cache.amazonaws.com'    
    action_params['redis_port'] = 6379
    #simple_invokation(action_params,256)

    populate_cache_until_full(action_params,256,'cachet2micro_32_4096mb_ec2_with_data_transfer.csv',300)
 