# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import data_feed_pb2 as data__feed__pb2


class DatasetFeedStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.shutdown = channel.unary_unary(
        '/DatasetFeed/shutdown',
        request_serializer=data__feed__pb2.Empty.SerializeToString,
        response_deserializer=data__feed__pb2.Empty.FromString,
        )
    self.registerjob = channel.unary_unary(
        '/DatasetFeed/registerjob',
        request_serializer=data__feed__pb2.RegisterJobRequest.SerializeToString,
        response_deserializer=data__feed__pb2.RegisterJobResponse.FromString,
        )
    self.getBatches = channel.unary_unary(
        '/DatasetFeed/getBatches',
        request_serializer=data__feed__pb2.GetBatchesRequest.SerializeToString,
        response_deserializer=data__feed__pb2.Sample.FromString,
        )


class DatasetFeedServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def shutdown(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def registerjob(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def getBatches(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DatasetFeedServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'shutdown': grpc.unary_unary_rpc_method_handler(
          servicer.shutdown,
          request_deserializer=data__feed__pb2.Empty.FromString,
          response_serializer=data__feed__pb2.Empty.SerializeToString,
      ),
      'registerjob': grpc.unary_unary_rpc_method_handler(
          servicer.registerjob,
          request_deserializer=data__feed__pb2.RegisterJobRequest.FromString,
          response_serializer=data__feed__pb2.RegisterJobResponse.SerializeToString,
      ),
      'getBatches': grpc.unary_unary_rpc_method_handler(
          servicer.getBatches,
          request_deserializer=data__feed__pb2.GetBatchesRequest.FromString,
          response_serializer=data__feed__pb2.Sample.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'DatasetFeed', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
