import grpc
from concurrent import futures
import data_feed_pb2
import data_feed_pb2_grpc
import multiprocessing as mp
import logging
import sys
from job import MLTrainingJob
from typing import Dict

# Logging initialization
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class DatasetFeedService(data_feed_pb2_grpc.DatasetFeedServicer):

    def __init__(self, kill_event,args):
        from coordinator import DataFeedCoordinator
        self.args = args
        self.coordinator = DataFeedCoordinator(self.args)
        self.training_jobs:Dict[str, MLTrainingJob] = {}
        self.kill_event = kill_event

    def shutdown(self, request, context):
        logger.info("Received shutdown request - Not implemented")
        # from main_grpc_client import shutdown_data_service
        # shutdown_data_service()
        context.set_code(grpc.StatusCode.OK)
        context.set_details('Shutting down')
        return data_feed_pb2.Empty()
        
    def registerjob(self, request, context):
        job_id = request.job_id
        if job_id in self.training_jobs:
            return data_feed_pb2.MessageResponse(message="Job {} already exists. Please register with a different job id".format(job_id))
        else:
            self.training_jobs[job_id] = MLTrainingJob(job_id=job_id, 
                                                   coordinator=self.coordinator,
                                                   args=self.args)
            self.training_jobs[job_id].start_data_prep_workers()
            return data_feed_pb2.MessageResponse(message="Job {} successfully regsistered".format(job_id))
    
    def getBatches(self, request, context):
        job_id = request.job_id
        batch_data, batch_id = self.training_jobs[job_id].next_batch()
        return data_feed_pb2.Sample(batchid=batch_id,
                                    data = batch_data,
                                    label = str(batch_id))

def start(kill_event, args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_feed_pb2_grpc.add_DatasetFeedServicer_to_server(
        DatasetFeedService(kill_event,args), server)
    server.add_insecure_port("[::]:" + args.port)
    server.start()
    logger.info("gRPC Data Server started, listening on port {}.".format(args.port))
    return server

def shutdown(grpc_server):
    logger.info('Shutting down...')
    logger.info('Stopping gRPC server...')
    grpc_server.stop(2).wait()
    logger.info('Shutdown done.')
    import os, time
    os.system('kill -9 %d' % os.getpid())

def wait_for_shutdown_signal():
    SHUTDOWN_PORT = 16000
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', SHUTDOWN_PORT))
    s.listen(1)
    #logger.info('Awaiting shutdown signal on port {}'.format(SHUTDOWN_PORT))
    conn, addr = s.accept()
    print('Received shutdown signal from: ', addr)
    try:
        conn.close()
        s.close()
    except Exception as e:
        logger.info(e)
        
def serve(args):
    kill_event = mp.Event() # an mp.Event for graceful shutdown
    grpc_server = start(kill_event, args)
    wait_for_shutdown_signal()
    kill_event.set()
    shutdown(grpc_server)


def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="input batch size for training",)
    
    parser.add_argument("--s3-bucket", type=str,default='sdl-cifar10',help="port for gprc server")
    parser.add_argument("--redis-host", type=str,default='test.rdior4.ng.0001.usw2.cache.amazonaws.com',help="port for gprc server")
    parser.add_argument("--redis-port", type=str,default='6379',help="port for gprc server")
    parser.add_argument("--lambda_func_name", type=str,default='lambda_dataloader_pytorch',help="port for gprc server")

    parser.add_argument("--port", type=str,default='50052',help="port for gprc server")
    parser.add_argument("--grpc-workers", type=int, default=1, metavar="N",
                        help="No. of gRPC server workers",)
    
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    serve(read_args())