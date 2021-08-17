from __future__ import print_function
import logging
import argparse
import grpc
import prac__pb2
import prac__pb2_grpc

parser = argparse.ArgumentParser(description='grpc Training')

parser.add_argument('-m', '--mode', default='r', type=str,
                    help='mode : c : create, r : read, u : update , d : delete')

parser.add_argument('-k', '--key', default=None, type=str,
                    help='key : the list to add, delete or be succeed')

def run():
    args = parser.parse_args()
    if args.mode == 'c':
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = prac__pb2_grpc.TODO_APPStub(channel)
            response = stub.Create_TD(prac__pb2.TD_create(Todo=args.key))
            print("received: " + response.message)

    elif args.mode == 'r':
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = prac__pb2_grpc.TODO_APPStub(channel)
            response = stub.Read_TD(prac__pb2.TD_read(Todo=args.key))
            print("received: " + response.list)

    elif args.mode == 'u':
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = prac__pb2_grpc.TODO_APPStub(channel)
            response = stub.Update_TD(prac__pb2.TD_update(Todo=args.key))
            print("received: " + response.message)

    elif args.mode == 'd':
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = prac__pb2_grpc.TODO_APPStub(channel)
            response = stub.Remove_TD(prac__pb2.TD_remove(Todo=args.key))
            print("received: " + response.message)


if __name__ == "__main__":
    logging.basicConfig()
    run()