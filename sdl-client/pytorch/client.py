import grpc
import cms_pb2_grpc as pb2_grpc
import cms_pb2 as pb2
import json

class CMSClient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50052
        self.job_avg_training_speed =0
        self.prev_batch_dl_delay =0
        self.avg_delay_on_miss = 0
        self.avg_delay_on_hit = 0.08
        # instantiate a channe
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.CacheManagementServiceStub(self.channel)

    def record_training_stats(self, avg_speed, dl_delay, cache_hit):

        if cache_hit:
            self.avg_delay_on_hit = dl_delay
        else:
            self.avg_delay_on_miss = dl_delay
        self.job_avg_training_speed = avg_speed
        #self.prev_batch_dl_delay = dl_delay
    
    def job_ended_nofifcation(self,job_id):
   
        messageRequest = pb2.JobEndedMessage(job_id=job_id)
        response = self.stub.ProcessJobEndedMessage(messageRequest)
        print(response.message)

    def register_training_job(self,job_id:int,batch_size:int):
        """
        Client function to call register a new ML training job
        """
        messageRequest = pb2.RegisterTrainingJobMessage(job_id=job_id,batch_size=batch_size)
        response = self.stub.RegisterNewTrainingJob(messageRequest)
        job_registered = response.registered
        if job_registered:
            labelled_dataset = json.loads(response.dataset)
        else:
            labelled_dataset = None
        return response.message, job_registered,labelled_dataset, response.batches_per_epoch
    
    def get_next_batch_for_job(self,job_id):
        """
        Client function to get next batch for ML training job
        """
        messageRequest = pb2.GetNextBatchForJobMessage(job_id=job_id,avg_training_speed =self.job_avg_training_speed, avg_delay_on_miss=self.avg_delay_on_miss, avg_delay_on_hit = self.avg_delay_on_hit)
        response = self.stub.GetNextBatchForJob(messageRequest)
        return  (response.batch_id, json.loads(response.batch_metadata), response.isCached, response.batch_data)
    
    def get_url(self, message):
        """
        Client function to call the rpc for GetServerResponse
        """
        message = pb2.Message(message=message)
        print(f'{message}')
        return self.stub.GetServerResponse(message)
    

#functions for testing the implementation

def test_register_new_job(cleint:CMSClient):
    result = client.register_training_job(job_id=1,batch_size=128)
    message = result.message
    registered = result.registered
    if registered:
        dataset = json.loads(result.dataset)
    else:
        dataset = None
    print(message,registered)

if __name__ == '__main__':
    client = CMSClient()
    test_register_new_job(cleint=client)
    #result = client.get_url(message="Hello Server you there?")
    #print(f'{result}')
    