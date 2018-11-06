from __future__ import print_function

import argparse
import time
import numpy as np

from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def run(host, port, model, signature_name):

    # channel = grpc.insecure_channel('%s:%d' % (host, port))
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Read an image
    #data = imread(image)
    #data = data.astype(np.float32)
    #print(data)
    features = np.array([5,1629097,0], dtype=np.int64)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['hash_imei'].CopyFrom(make_tensor_proto(features[0], shape=[1, 1]))
    request.inputs['app_id'].CopyFrom(make_tensor_proto(features[1], shape=[1, 1]))
    request.inputs['as_gs'].CopyFrom(make_tensor_proto(features[2], shape=[1, 1]))

    print("===========")
    print(request)
    print("===========")
    
    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='localhost', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    #parser.add_argument('--features', help='input image', type=str)
    parser.add_argument('--model', help='model name', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='serving_default', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.model, args.signature_name)
