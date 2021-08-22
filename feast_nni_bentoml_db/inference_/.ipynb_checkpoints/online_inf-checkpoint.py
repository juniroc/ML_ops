# -*- coding: utf-8 -*-
# +
import requests
import feast
import pandas as pd
from feast import Entity, Feature, FeatureView, FileSource, ValueType
import torch
import pymysql
from datetime import date
from sqlalchemy import create_engine
import numpy as np
pymysql.install_as_MySQLdb()
import MySQLdb
import math

import warnings
warnings.filterwarnings("ignore")


# -

def get_entities(path_,from_time_):
    parquet_ = pd.read_parquet(path_, engine='pyarrow')
    orders = parquet_[['ticket_id','event_timestamp']]

    ### 뽑을 rows의 entity == key 라고 볼 수 있음
    new_orders = orders[orders['event_timestamp']>= from_time_]
    new_orders.drop_duplicates(ignore_index=True, inplace=True)
    return new_orders


def get_df_from_online(feast_path_, entity_df_):
    fs_ = feast.FeatureStore(repo_path=feast_path_)

    online_ = fs_.get_online_features(
                entity_rows=[{"ticket_id": i} for i in entity_df_['ticket_id']],
                features=[
                    "dr_lauren_stat:time",
                    "dr_lauren_stat:weekday",
                    "dr_lauren_stat:weekend",
                    "dr_lauren_stat:instlo_1",
                    "dr_lauren_stat:instlo_2",
                    "dr_lauren_stat:inst_code",
                    "dr_lauren_stat:sysname_lo",
                    "dr_lauren_stat:sysname_eq",
                    "dr_lauren_stat:ntt_label",
            ],
    )


    df = pd.DataFrame.from_dict(online_.to_dict())

    return df[['time', 'weekday', 'weekend', 'instlo_1', 'instlo_2', 'inst_code', 'sysname_lo', 'sysname_eq','ntt_label']]


# ### request

# +
def get_from_bentoml(online_df_):
    response = requests.post("http://127.0.0.1:5000/predict", data=online_df_.iloc[:,:-1].to_json())
    infer_ = response.text
    predict_ = infer_[1:-1].split(', ')
    return predict_

def get_infer_df_(entity_df_, online_df_):
    infer_df = entity_df_.reset_index(drop=True)
    infer_df['label'] = online_df_['ntt_label']
    infer_df['predict'] = get_from_bentoml(online_df_)
    return infer_df


# +
### data create, insert, update
def get_commit_sql(conn_, sql_):
    with conn_.cursor() as cursor:
        cursor.execute(sql_)
        result = cursor.fetchall()
        conn.commit()

### data read
def get_data_sql(conn_, sql_):
    with conn_.cursor() as cursor:
        cursor.execute(sql_)
        result = cursor.fetchall()
        return result


# -

# ### create table

# +
# sql = '''
#     CREATE TABLE infer_df (
#         ticket_id int(11) NOT NULL AUTO_INCREMENT PRIMARY KEY,
#         event_timestamp varchar(255) NOT NULL,
#         label int(11) NOT NULL,
#         predict int(11) NOT NULL
#     ) ENGINE=InnoDB DEFAULT CHARSET=utf8
# '''
# get_commit_sql(conn, sql)
# -

def get_columns_(df_):
    columns_str = ', '.join([i for i in infer_df.columns])
    return columns_str


def get_values_(df_):
    value_ = []
    for row_ in df_.values:
        value_.append("("+", ".join(["'"+str(i)+"'" for i in row_]) + ")")

    values_str = ', '.join(value_)
    return values_str


def pred_into_db(conn_, infer_df_):
    columns_ = get_columns_(infer_df_)

    values_ = get_values_(infer_df_)

    sql = f"INSERT INTO infer_df ({columns_}) VALUES {values_} ;"
    get_commit_sql(conn_, sql)


def main():
    path = 'workspace/ML_Ops/feast_fea_/data/ppr_data_.parquet'
    from_time = '2021-07-01'

    entity_df = get_entities(path, from_time)
    
    feast_path = './'
    online_df = get_df_from_online(feast_path, entity_df)
    
    infer_df = get_infer_df_(entity_df, online_df)
    
    conn = pymysql.connect(host='localhost', port=3306, user='root', bind_address='127.0.0.1', passwd='', db='infer_db', charset='utf8')
    
    pred_into_db(conn,infer_df)


main()
