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

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.CacheManagementServiceStub(self.channel)
    
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
    