# gRPC 

## Reference

1. 파이썬 gRPC Tutorial
    * https://grpc.io/docs/languages/python/quickstart/

2. Buf - protobuf 파일을 빌드시켜주는 툴
    * https://github.com/bufbuild/buf
    * Python 빌드 방법: https://github.com/bufbuild/buf/issues/223
    * 공식 문서_1 : https://developers.google.com/protocol-buffers/docs/tutorials
    * 공식 문서_2 : https://github.com/protocolbuffers/protobuf/tree/master/examples

3. BloomRPC
    * gRPC를 GUI에서 할 수 있도록 하는 툴
    * https://github.com/uw-labs/bloomrpc 


## 구현 list

**TODO 애플리케이션 작성**

1. 할일 등록 (Create) => CreateTodo
2. 할일 삭제 (Update) => Clo seTodo
3. 할일 완료 (Update) => SuccessTodo
4. 할일 리스트 가져오기 (Read) => ListTodo



---


### practice

**0. `prac_.proto` proto_file 생성**
* 각 service instance에 맞춰, message 생성


`prac_.proto` 
```
syntax = "proto3";

package practice;

service TODO_APP {
  // Create TODO_
  rpc Create_TD (TD_create) returns (complete_C);
  // Remove TODO_
  rpc Remove_TD (TD_remove) returns (complete_D);
  // finish TODO_
  rpc Update_TD (TD_update) returns (complete_U);
  // Read TODO_
  rpc Read_TD (TD_read) returns (complete_R);
}


// the Create ToDo

### create service
message TD_create {
  string Todo = 1;
}

message complete_C {
  string message = 1;
}

---

### remove service
message TD_remove {
  string Todo = 1;
}

message complete_D {
  string message = 1;
}

---

### update service
message TD_update {
  string Todo = 1;
}

message complete_U {
  string message = 1;
}

---

### read service
message TD_read {
  string Todo = 1;
}

message complete_R {
  string list = 1;
}

```


---


**1. 위 생성된 `prac_.proto` proto_file 실행**

```
python -m grpc_tools.protoc -I. --python_out=./python_ --grpc_python_out=./python_ prac_.proto
```

* `output_file` 위치 입력 `prac_.proto` 파일 실행
* `prac_pb2__grpc.py`, `prac__pb2.py` 이라는 `output_file` 생성
* 이후에 `server.py` 을 작성시 request 형식을 prac_pb2.py 아래에 있는 객체 양식에 맞춰줘야 함
* `client.py` 작성시 `prac__pb2_grpc.py`의 stub 과 `prac__pb2.py`의 객체 양식 또한 맞춰줘야 함

*사실상 여기서 생성된 파일들은 내부 구조만 파악하고 건드릴 것은 없음*



![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/gRPC_/capture/0.PNG)



---



**2. `TOdo_server.py` 와 `Todo_client.py` file 작성**

* 객체내 메소드 작성시 request 타입을 정할 수 있음
* dataset은 따로 생성해둔 csv 파일 이용
* 딕셔너리 형식을 이용해 csv 파일을 key:value 쌍으로 묶는다.
* argparse 라이브러리를 이용해 cli 명령어 입력시 `원하는 모드 : <c,r,u,d>`와 `원하는리스트 : key` 받아옴 


* Create_TD 
    -> 이미 리스트에 존재하는 경우 존재 메시지 출력
    -> 없는 경우 0 으로 할당하여 key:value 값 할당
    -> 실행했을 경우 1 안했을 경우 0


* Remove_TD
    -> 존재하는 경우 제거
    -> 존재하지 않는 경우 메시지 출력


* Update_TD
    -> 존재하는 경우 0->1 로 변환
    -> 존재하지 않는 경우 메시지 출력

* Read_TD
    -> key 값으로 검색
    -> value 값이 1인 경우 : Suceess_`key`
    -> value 값이 0인 경우 : Not_yet_'key`


`Todo_server.py`
```
import pandas as pd
from concurrent import futures
import logging

import grpc

import prac__pb2
import prac__pb2_grpc


## 객체내 메소드 작성시 request 의 타입을 미리 정할 수 있음

class TODO_APP(prac__pb2_grpc.TODO_APPServicer):

    def Create_TD(self, request : prac__pb2.TD_create, context):

        df = pd.read_csv('./datas/todo_lst.csv', index_col='name')
        dict_ = df.to_dict(orient='dict')
        dict_ = dict_['value']


        ## todo_lst 에 이미 존재할 경우 
        if request.Todo in dict_.keys():
            return prac__pb2.complete_C(message=f'It already have {request.Todo}')

        ## 그렇지 않을 경우 생성해주고 진행되지 않았으므로 '0' 으로 할당
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

```


`Todo_client.py`
```
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

```



---



**3. cli 환경에서 server 띄우기**

```
python Todo_server.py
```



---



**4. 다른 cli 환경을 열어 요청보내기**

```
python Todo_client.py -m `<c : 원하는 모드>` -k `<추가할 리스트>`
```


\
**test_capture**
\
\
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/gRPC_/capture/1.PNG)

\
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/gRPC_/capture/2.PNG)

\
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/gRPC_/capture/3.PNG)
