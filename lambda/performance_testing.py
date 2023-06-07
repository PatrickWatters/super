
from super_lambda_mgmt import SUPERLambdaMgmt
import random
import time
def gen_dummy_batch(batch_size=256):
    batch_metadata = []
    for i in range(0,batch_size):
        batch_metadata.append(('test/Frog/leopard_frog_s_001876.png',6))
    return batch_metadata

def simple_invokation():
    mgr = SUPERLambdaMgmt()
    batch_metadata = gen_dummy_batch(128)
    action_params = {}
    action_params['batch_metadata'] = batch_metadata
    action_params['batch_id'] = random.randint(0,10000000000)
    action_params['cache_bacth'] = True
    action_params['return_batch_data'] = False
    action_params['bucket'] = 'sdl-cifar10'    
    action_params['redis_host'] = 'redistest.rdior4.ng.0001.usw2.cache.amazonaws.com'    
    action_params['redis_port'] = '6379'

    end = time.time()
    #call function
    response = mgr.invoke_function(action_params, None)
    print(response)
    print(time.time() - end)
    
if __name__ == '__main__':
     simple_invokation()