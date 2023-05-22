from logging import error
from PIL import Image
import redis
import io
import time
import argparse

class RedisClient:
    def __init__(self, host='sion'):
        self.exire_time = 0
        if host == 'local':
            endpoint = '127.0.0.1'
            self.conn = redis.StrictRedis(host=endpoint, port=6379)
            self.exire_time = 10
        elif host == 'aws':
            endpoint='example-001.xjfut6.0001.usw1.cache.amazonaws.com'
            self.conn = redis.StrictRedis(host=endpoint, port=6379)
        else:
            host = 'sion'
            #endpoint = '127.0.0.1'
            endpoint = '34.218.238.210' #public ip address - works for testing/debugging locally:)
            self.conn = redis.StrictRedis(host=endpoint, port=6378)
        self.host = host
        
    def insert_data(self, key, value):
        try:
            if self.host == 'sion':
                self.conn.set(key, value)
                return True, None
            else:
                #self.conn.set(key, value,ex=self.exire_time)
                self.conn.set(key, value)
                return True, None
            
        except Exception as exception:
            #print(exception)
            return False, exception
            #return  None

    def get_data(self, key):
        try:
            return self.conn.get(key)
        except Exception as exception:
            return  None
    
    def empty_db(self):
        try:
            self.conn.flushdb()
        except Exception as exception:
            return  None
            
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('-flush', '--flush', default=False, type=bool, metavar='N', help='Empties the cache if TRUE')
    parser.add_argument('-host', '--host', default='local', type=str,metavar='N')   
    args = parser.parse_args()

    if args.host == 'sion':
        r = RedisClient(host = args.host)
        data = r.insert_data('v1',1000)
        dataget = r.get_data('v1')
        print(dataget)
    else:
        r = RedisClient(host = args.host)
        #r.conn.expire("b1",time=10)
        r.insert_data('v1',1000)
        if(args.flush == True):
            print('Item count before flush:',r.conn.dbsize())
            r.conn.flushdb()
            print('Item count after flush',r.conn.dbsize())
        else:
            print('Item Count:',r.conn.dbsize())

'''
if __name__ == "__main__":
    r = RedisClient(host='sion')
    data = r.insert_data('t2',1000)
    dataget = r.get_data('t2')
    print(dataget)
    #r.conn.flushdb()  
    #print('DB Flushed')
'''
    
