"""
Example DAG demonstrating the usage of the TaskFlow API to execute Python functions natively and within a
virtual environment.
"""
import logging
import shutil
import time
import pendulum
from pprint import pprint
from docker.types import Mount


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


preprocessing = DockerOperator(task_id="load_and_preprocessing_train",
                                image = "lmj3502/airflow_gnn",
                                command=["python", "/gnn_py/preprocessing.py", "--config-path", '/gnn_py/config_files/config_train.yml'],
                                mounts=[
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/config_files", target ="/gnn_py/config_files", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/dataset/", target="/mnt/", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/versions/", target="/gnn_py/versions/", type="bind"),
                                ],
                            dag=dag,
                            )



create_graph = DockerOperator(task_id="create_graph_train",
                                image = "lmj3502/airflow_gnn",
                                command=["python", "/gnn_py/create_graph.py", "--config-path", '/gnn_py/config_files/config_train.yml'],
                                mounts=[
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/config_files", target ="/gnn_py/config_files", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/dataset/", target="/mnt/", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/versions/", target="/gnn_py/versions/", type="bind"),
                                ],
                            dag=dag,
                            )

training = DockerOperator(task_id="training",
                                image = "lmj3502/airflow_gnn",
                                command=["python", "/gnn_py/training.py", "--config-path", '/gnn_py/config_files/config_train.yml'],
                                mounts=[
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/config_files", target ="/gnn_py/config_files", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/dataset/", target="/mnt/", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/versions/", target="/gnn_py/versions/", type="bind"),
                                ],
                            dag=dag,
                            )



preprocessing_inf = DockerOperator(task_id="load_and_preprocessing_inf",
                                image = "lmj3502/airflow_gnn",
                                command=["python", "/gnn_py/preprocessing.py", "--config-path", '/gnn_py/config_files/config_inf.yml'],
                                mounts=[
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/config_files", target ="/gnn_py/config_files", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/dataset/", target="/mnt/", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/versions/", target="/gnn_py/versions/", type="bind"),
                                ],
                            dag=dag,
                            )


create_graph_inf = DockerOperator(task_id="create_graph_inf",
                                image = "lmj3502/airflow_gnn",
                                command=["python", "/gnn_py/create_graph.py", "--config-path", '/gnn_py/config_files/config_inf.yml'],
                                mounts=[
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/config_files", target ="/gnn_py/config_files", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/dataset/", target="/mnt/", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/versions/", target="/gnn_py/versions/", type="bind"),
                                ],
                            dag=dag,
                            )

inference = DockerOperator(task_id="inference",
                                image = "lmj3502/airflow_gnn",
                                command=["python", "/gnn_py/inference.py", "--config-path", '/gnn_py/config_files/config_inf.yml'],
                                mounts=[
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/config_files", target ="/gnn_py/config_files", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/dataset/", target="/mnt/", type="bind"),
                                    Mount(source="/home/juniroc/ml-workspace/airflow_gnn/versions/", target="/gnn_py/versions/", type="bind"),
                                ],
                            dag=dag,
                            )


preprocessing >> create_graph >> training >> preprocessing_inf >> create_graph_inf >> inference