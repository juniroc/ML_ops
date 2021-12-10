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
