# 데이터 ETI 
- Extract
- Transform
- Inference
-----

### airflow install 후 WebUI 확인
`airflow webserver --port 8080\`

`localhost:8080` 으로 접속
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_1.png)


### dag 추가하기
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_2.png)

`~/airflow/` 에서 `airflow.cfg` 파일 확인

- `dags_folder` 라는 것을 확인할 수 있음
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_3.png)

- `dags` 디렉토리안에 파이썬 파일을 넣고 실행
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_4.png)

- `airflow dags list`를 통해 dags 목록이 생성된것을 확인
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_5.png)

- 그중 내가 만든 `test_1`
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_6.png)

- `airflow tasks list test_1` 를 통해 생성된 dag의 tasks 리스트 출력
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_7.png)

### web UI dag 등록 확인 및 실행
- `airflow scheduler`를 이용해 등록 확인
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_8.png)

그리고 만약 `Not yet start` 인 상태로 멈춰있으면

`airflow scheduler` 명령어 입력해주면됨
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_9.png)

잘 작동함을 확인할 수 있음
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_10.png)


### airflow_2 version 
- `decorator`를 통해 컴포넌트 생성 가능

- `data Extract` -> `data Transform` -> `data Inference` 진행하는 코드

`ETI_test.py`
```
import pandas as pd
import os
import json

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

from pipeline_.preprocess_ import preprocess

from pipeline_.models_ import models_ as mo_

from scipy.stats import entropy

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'juniroc',
}

@dag('dr_lauren_epi', default_args=default_args, schedule_interval='* * * * *', start_date=days_ago(1), tags=['dr_lauren'])
def taskflow_extract_ppr_infer():
    """
    dr_lauren_project dataset을 이용해 데이터 추출 - 전처리 - 추론 과정 
    """
    @task()
    def extract_1():
        """
        data Extract
        """
        inf_path_ = '/home/zeroone/airflow/dags/dataset_/infer_df.csv'
        trained_path_ = '/home/zeroone/airflow/dags/dataset_/210818_2.csv'


        infer_df = pd.read_csv(inf_path_, index_col=0).iloc[:500,:]
        # trained_df = pd.read_csv(trained_path_, index_col=0)

        infer_str = infer_df.to_json(orient="columns")
        # trained_str = trained_df.to_json(orient="columns")
        

        # infer_parse = json.loads(infer_str)
        # trained_parse = json.loads(trained_str)

        # print(type(infer_parse))

        return infer_str

    @task()
    def extract_2():
        """
        data Extract
        """
        inf_path_ = '/home/zeroone/airflow/dags/dataset_/infer_df.csv'
        trained_path_ = '/home/zeroone/airflow/dags/dataset_/210818_2.csv'


        # infer_df = pd.read_csv(inf_path_, index_col=0).iloc[:500,:]
        trained_df = pd.read_csv(trained_path_, index_col=0)

        # infer_str = infer_df.to_json(orient="columns")
        trained_str = trained_df.to_json(orient="columns")
        

        # infer_parse = json.loads(infer_str)
        # trained_parse = json.loads(trained_str)

        # print(type(infer_parse))

        return trained_str


    @task()
    def preprocessing_(inf_, trnd_):

        """
        #### preprocess task
        preprocessing for inference.
        """

        print('여기부터 preprocessing')
        print(inf_)
        
        print(trnd_)
        print('infer_df print_')

        inf_df = pd.read_json(inf_)
        trnd_df = pd.read_json(trnd_)
        
        print(inf_df)
        print(trnd_df)        

        ppr = preprocess(inf_df, trnd_df)
        
        # inf_df = ppr.ppr_2(inf_df)
        inf_df.drop_duplicates(ignore_index=True, inplace=True)
        inf_df = ppr.ppr_le(inf_df, trnd_df)
        print('final_inf_df')
        print(inf_df)
    

        ppred_str = inf_df.to_json(orient="columns")

        return ppred_str

    @task()
    def inference(ppred_str):
        """
        #### inference task and return values
        A simple Load task which takes in the result of the Transform task and
        instead of saving it to end user review, just prints it out.
        """

        model_path = '/home/zeroone/airflow/dags/pipeline_/models_/lgb_model.pth.tar'

        m_ = mo_(model_path)

        ppred_df = pd.read_json(ppred_str)

        result_ = m_.get_predict(ppred_df)
        prob_ = m_.get_proba(ppred_df)

        final_ = ppred_df[['ticket_id', 'fault_time']]
        final_['prob_'] = prob_[:,1]
        final_['entropy_'] = entropy([prob_[:,0], prob_[:,1]], base=2)

        print(final_)



        
    inf_ = extract_1()
    print(inf_)
    trnd_ = extract_2()
    print(trnd_)

    transformed_df = preprocessing_(inf_,trnd_)
    
    inference(transformed_df)



tutorial_etl_dag = taskflow_extract_ppr_infer()
```

아래와 같이 실험 과정 성공 및 실패 여부를 알 수 있음 
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_11.png)

- 정상적으로 돌아갔을 경우 결과
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_12.png)


### schedule 변수 설정
- `schedule_interval` 분단위 시단위 일 주 월 단위로 지정할 수 있으며, 초단위로도 가능
- `start_data` 를 통해 (처리할 데이터의) 시작 날짜를 지정할 수 있음 
- `tag` 를 통해 실험 구분 가능 및 버져닝
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_13.png)

### 최종 결과 확인
![image](https://github.com/juniroc/ML_ops/blob/main/Air_flow/image/image_14.png)
