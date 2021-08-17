import pandas as pd
from concurrent import futures
import logging

import grpc

import prac__pb2
import prac__pb2_grpc


class TODO_APP(prac__pb2_grpc.TODO_APPServicer):

    def Create_TD(self, request : prac__pb2.TD_create, context):

        df = pd.read_csv('./datas/todo_lst.csv', index_col='name')
        dict_ = df.to_dict(orient='dict')
        dict_ = dict_['value']

        if request.Todo in dict_.keys():
            return prac__pb2.complete_C(message=f'It already have {request.Todo}')

        else:
            dict_[request.Todo] = 0
            df = pd.DataFrame.from_dict(dict_, orient='index', columns=['value'])
            df.to_csv('./datas/todo_lst.csv', index_label='name')

            return prac__pb2.complete_C(message=f'append_{request.Todo}')

    def Remove_TD(self, request : prac__pb2.TD_remove, context):

        df = pd.read_csv('./datas/todo_lst.csv', index_col='name')
        dict_ = df.to_dict(orient='dict')
        dict_ = dict_['value']

        if request.Todo in dict_.keys():
            del dict_[request.Todo]
            df = pd.DataFrame.from_dict(dict_, orient='index', columns=['value'])
            df.to_csv('./datas/todo_lst.csv', index_label='name')

            return prac__pb2.complete_D(message=f'deleted_{request.Todo}')

        else:
            return prac__pb2.complete_D(message=f'TODO_list does not have {request.Todo}')

    def Update_TD(self, request : prac__pb2.TD_update, context):

        df = pd.read_csv('./datas/todo_lst.csv', index_col='name')
        dict_ = df.to_dict(orient='dict')
        dict_ = dict_['value']

        if request.Todo in dict_.keys():
            dict_[request.Todo] = 1
            df = pd.DataFrame.from_dict(dict_, orient='index', columns=['value'])
            df.to_csv('./datas/todo_lst.csv', index_label='name')

            return prac__pb2.complete_U(message=f'Success_{request.Todo}')

        else:
            return prac__pb2.complete_U(message=f'there is no {request.Todo}')

    def Read_TD(self, request : prac__pb2.TD_read, context) :

        df = pd.read_csv('./datas/todo_lst.csv', index_col='name')
        dict_ = df.to_dict(orient='dict')
        dict_ = dict_['value']

        Succeed_lst = [key for key, value in dict_.items() if value ==1]
        not_yet_lst = [key for key, value in dict_.items() if value ==0]

        return prac__pb2.complete_R(list=f'Succeed_{",".join(Succeed_lst)},\n Not_yet_{",".join(not_yet_lst)}')

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prac__pb2_grpc.add_TODO_APPServicer_to_server(TODO_APP(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()