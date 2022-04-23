"""
Example DAG demonstrating the usage of the TaskFlow API to execute Python functions natively and within a
virtual environment.
"""
import logging
import shutil
import time
from pprint import pprint
from docker.types import Mount

import pendulum

from airflow import DAG
from airflow.decorators import task
from airflow.providers.docker.operators.docker import DockerOperator

log = logging.getLogger(__name__)

dag = DAG(
    dag_id='test_Docker_Operator',
    schedule_interval=None,
    start_date=pendulum.datetime(2022, 4, 19, tz="UTC"),
    tags=['test'],
)
    # [START howto_operator_python]


preprocessing = DockerOperator(task_id="load_and_preprocessing",
                                image = "lmj3502/airflow_gnn",
                                # command=["sh", "./main.sh", "train", "preprocess"],
                                command=["preprocessing.py", "--config-path", './config_files/config_train.yml'],
                                mounts=[
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/config_files", target ="/gnn_py/config_files", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/dataset/", target="/mnt/", type="bind"),
                                ],
                            # volumes=['/home/juniroc/ml-workspace/airflow_gnn/config_files:/gnn_py/config_files'],
                            # volumes=["/home/juniroc/ml-workspace/airflow_gnn/dataset/:/mnt/"],
                            dag=dag,
#                             network_mode='bridge'
                            )


preprocessing
