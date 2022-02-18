# Training_2 : kfp-caip-sklearn

**목표**

1. Kuberflow 사전 빌드 구성 요소를 사용하는 방법
2. bigquery, AI 플랫폼, 학습, 예측. Clip Flow 경량 Python 구성 요소를 사용하는 방법

# Continuous training pipeline with Kubeflow Pipeline and AI Platform

→ The workflow implemented by the pipeline is defined using a Python based Domain Specific Language (DSL). The pipeline's DSL is in the `covertype_training_pipeline.py` file that we will generate below.

The pipeline's DSL has been designed to avoid hardcoding any environment specific settings like file paths or connection strings. These settings are provided to the pipeline code through a set of environment variables.

[trainer_image/Dockerfile](Training_2%202412c/trainer_im%205d6fe.md)

[trainer_image/train.py](Training_2%202412c/trainer_im%20c9810.md)

[base_image/Dockerfile](Training_2%202412c/base_image%201f786.md)

### Pipeline 은 pre-build and Custom components 를 이용

### 1) Pre-build components

1. **BigQuery query component**
    
    [kubeflow/pipelines](https://github.com/kubeflow/pipelines/tree/0.2.5/components/gcp/bigquery/query)
    
    - Select training data by submitting a query to BigQuery.
    - Output the training data into a Cloud Storage bucket as CSV files.
    
2. **AI Platform Training component**
    
    [kubeflow/pipelines](https://github.com/kubeflow/pipelines/tree/0.2.5/components/gcp/ml_engine/train)
    
    - Use this component to submit a training job to AI Platform from a Kubeflow pipeline.
    
3. **AI Platform Deploy component**
    
    [kubeflow/pipelines](https://github.com/kubeflow/pipelines/tree/0.2.5/components/gcp/ml_engine/deploy)
    
    - Use the component to deploy a trained model to Cloud ML Engine. The deployed model can serve online or batch predictions in a Kubeflow Pipeline.

### 2) Custom components

→ `helper_components.py` file에 존재

[helper_components.py](Training_2%202412c/helper_com%209f0bf.md)

1. **Retrieve Best Run**
    - This component retrieves a tuning metric and hyperparameter values for the best run of a AI Platform Training hyperparameter tuning job.
2. **Evaluate Model**
    - This component evaluates a sklearn trained model using a provided metric and a testing dataset.

### Create Pipeline File

→ ./pipeline 에 `covertype_training_pipeline.py` 파일 생성 및 필요 라이브러리 호출

(1/5)

```python
%%writefile ./pipeline/covertype_training_pipeline.py
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KFP orchestrating BigQuery and Cloud AI Platform services."""

import os

from helper_components import evaluate_model
from helper_components import retrieve_best_run
from jinja2 import Template
import kfp
from kfp.components import func_to_container_op
from kfp.dsl.types import Dict
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret
```

환경변수 및 Hyperparameters Setting

(2/5)

```python
# Defaults and environment settings
BASE_IMAGE = os.getenv('BASE_IMAGE')
TRAINER_IMAGE = os.getenv('TRAINER_IMAGE')
RUNTIME_VERSION = os.getenv('RUNTIME_VERSION')
PYTHON_VERSION = os.getenv('PYTHON_VERSION')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')
USE_KFP_SA = os.getenv('USE_KFP_SA')

TRAINING_FILE_PATH = 'datasets/training/data.csv'
VALIDATION_FILE_PATH = 'datasets/validation/data.csv'
TESTING_FILE_PATH = 'datasets/testing/data.csv'

# Parameter defaults
SPLITS_DATASET_ID = 'splits'
HYPERTUNE_SETTINGS = """
{
    "hyperparameters":  {
        "goal": "MAXIMIZE",
        "maxTrials": 6,
        "maxParallelTrials": 3,
        "hyperparameterMetricTag": "accuracy",
        "enableTrialEarlyStopping": True,
        "params": [
            {
                "parameterName": "max_iter",
                "type": "DISCRETE",
                "discreteValues": [500, 1000]
            },
            {
                "parameterName": "alpha",
                "type": "DOUBLE",
                "minValue": 0.0001,
                "maxValue": 0.001,
                "scaleType": "UNIT_LINEAR_SCALE"
            }
        ]
    }
}
"""
```

Helper functions (data sampling)

→ 변수들을 통해 데이터를 **원하는 양식으로 sampling** 

(3/5)

```python

# Helper functions
def generate_sampling_query(source_table_name, num_lots, lots):
    """Prepares the data sampling query."""

    sampling_query_template = """
         SELECT *
         FROM 
             `{{ source_table }}` AS cover
         WHERE 
         MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(cover))), {{ num_lots }}) IN ({{ lots }})
         """
    query = Template(sampling_query_template).render(
        source_table=source_table_name, num_lots=num_lots, lots=str(lots)[1:-1])

    return query

```

컴포넌트 요소 셋팅

→ 1) Big query / Training / Deploying 컴포넌트
→ 2) Retrieve_best_run / Evaluate 컴포넌트

(4/5)

```python

# Create component factories
### TO DO list 
### Complete the command
### Use the pre-build bigquery/query component
### Use the pre-build ml_engine/train
### Use the pre-build ml_engine/deploy component
### Package the retrieve_best_run function into a lightweight component
### Package the evaluate_model function into a lightweight component

component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])
bigquery_query_op = component_store.load_component('bigquery/query')
mlengine_train_op = component_store.load_component('ml_engine/train')
mlengine_deploy_op = component_store.load_component('ml_engine/deploy')
retrieve_best_run_op = func_to_container_op(
    retrieve_best_run, base_image=BASE_IMAGE)
evaluate_model_op = func_to_container_op(evaluate_model, base_image=BASE_IMAGE)
```

위에서 정의한 함수 및 변수 이용해 파이프라인 구축, 평가 및 배포

1) Qeury sampling 함수를 이용 - train_set, val_set, test_set sampling

2) (Training Component 이용) HYPERTUNE_SETTINGS에 정의한 Hyperparameter 진행

3) Retrieve(1) / Evaluate Component(2) 이용해 Best_Value 추출

4) 기존 보다 높은 성능의 모델 배포

(5/5)

```python

@kfp.dsl.pipeline(
    name='Covertype Classifier Training',
    description='The pipeline training and deploying the Covertype classifierpipeline_yaml'
)
def covertype_train(project_id,
                    region,
                    source_table_name,
                    gcs_root,
                    dataset_id,
                    evaluation_metric_name,
                    evaluation_metric_threshold,
                    model_id,
                    version_id,
                    replace_existing_version,
                    hypertune_settings=HYPERTUNE_SETTINGS,
                    dataset_location='US'):
    """Orchestrates training and deployment of an sklearn model."""

### 1) data sampling

    # Create the training split
    query = generate_sampling_query(
        source_table_name=source_table_name, num_lots=10, lots=[1, 2, 3, 4])

    training_file_path = '{}/{}'.format(gcs_root, TRAINING_FILE_PATH)

    create_training_split = bigquery_query_op(
        query=query,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id='',
        output_gcs_path=training_file_path,
        dataset_location=dataset_location)

    # Create the validation split
    query = generate_sampling_query(
        source_table_name=source_table_name, num_lots=10, lots=[8])

    validation_file_path = '{}/{}'.format(gcs_root, VALIDATION_FILE_PATH)

    create_validation_split = bigquery_query_op(
        query=query,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id='',
        output_gcs_path=validation_file_path,
        dataset_location=dataset_location)

    # Create the testing split
    query = generate_sampling_query(
        source_table_name=source_table_name, num_lots=10, lots=[9])

    testing_file_path = '{}/{}'.format(gcs_root, TESTING_FILE_PATH)

    create_testing_split = bigquery_query_op(
        query=query,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id='',
        output_gcs_path=testing_file_path,
        dataset_location=dataset_location)

### 2) Hyperparameters

    # Tune hyperparameters
    tune_args = [
        '--training_dataset_path',
        create_training_split.outputs['output_gcs_path'],
        '--validation_dataset_path',
        create_validation_split.outputs['output_gcs_path'], '--hptune', 'True'
    ]

    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir/hypertune',
                                kfp.dsl.RUN_ID_PLACEHOLDER)

		### TO DO: Use the mlengine_train_op
### training component 이용해 hyper Tuning

    hypertune = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=tune_args,
        training_input=hypertune_settings)
    

### 3) 
### 3-1)Retrieve Component 이용해 Best_parameter 추출

    # Retrieve the best trial
    get_best_trial = retrieve_best_run_op(
            project_id, hypertune.outputs['job_id'])

### train, validation dataset 병합

    # Train the model on a combined training and validation datasets
    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir', kfp.dsl.RUN_ID_PLACEHOLDER)

    train_args = [
        '--training_dataset_path',
        create_training_split.outputs['output_gcs_path'],
        '--validation_dataset_path',
        create_validation_split.outputs['output_gcs_path'], '--alpha',
        get_best_trial.outputs['alpha'], '--max_iter',
        get_best_trial.outputs['max_iter'], '--hptune', 'False'
    ]

### 병합된 dataset 학습

### TO DO: Use the mlengine_train_op
    train_model = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=train_args)

### 3-2) test_set 으로 평가 
    # Evaluate the model on the testing split
    eval_model = evaluate_model_op(
        dataset_path=str(create_testing_split.outputs['output_gcs_path']),
        model_path=str(train_model.outputs['job_dir']),
        metric_name=evaluation_metric_name)

### 4) 더 높은 성능 model 배포

    # Deploy the model if the primary metric is better than threshold
    with kfp.dsl.Condition(eval_model.outputs['metric_value'] > evaluation_metric_threshold):
        deploy_model = mlengine_deploy_op(
        model_uri=train_model.outputs['job_dir'],
        project_id=project_id,
        model_id=model_id,
        version_id=version_id,
        runtime_version=RUNTIME_VERSION,
        python_version=PYTHON_VERSION,
        replace_existing_version=replace_existing_version
)

    # Configure the pipeline to run using the service account defined
    # in the user-gcp-sa k8s secret
    if USE_KFP_SA == 'True':
        kfp.dsl.get_pipeline_conf().add_op_transformer(
              use_gcp_secret('user-gcp-sa'))
```

return

Overwriting ./pipeline/covertype_training_pipeline.py

### Check the Docker file in base_image, trainer_image

```python
!cat base_image/Dockerfile
```

return

FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire scikit-learn==0.20.4 pandas==0.24.2 kfp==0.2.5

```python
!cat trainer_image/Dockerfile
```

return

FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire cloudml-hypertune scikit-learn==0.20.4 pandas==0.24.2
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]

## Building and Deploying the pipeline

- Before deploying to AI Platform Pipelines, the pipeline DSL has to be compiled into a pipeline runtime format, also refered to as a pipeline package. The runtime format is based on **Argo Workflow**, which is expressed in YAML.

[argoproj/argo-workflows](https://github.com/argoproj/argo-workflows)

### Getting Cloud Storage Bucket list

```python
!gsutil ls
```

return

gs://artifacts.qwiklabs-gcp-00-1af1e75d90f7.appspot.com/

gs://qwiklabs-gcp-00-1af1e75d90f7-kubeflowpipelines-default/

gs://qwiklabs-gcp-00-1af1e75d90f7_cloudbuild/

### Setting Env

- **How to Check ENDPOINT**
    
    **AI Platform** → **Pipelines → SETTINGS** 
    
    ![Training_2%202412c/Endpoint.png](Training_2%202412c/Endpoint.png)
    

```python
REGION = 'us-central1'
ENDPOINT = '7d74ffdf8b783b10-dot-us-central1.pipelines.googleusercontent.com' # TO DO: REPLACE WITH YOUR ENDPOINT
ARTIFACT_STORE_URI = 'gs://qwiklabs-gcp-00-1af1e75d90f7-kubeflowpipelines-default' # TO DO: REPLACE WITH YOUR ARTIFACT_STORE NAME 
PROJECT_ID = !(gcloud config get-value core/project)
PROJECT_ID = PROJECT_ID[0]

IMAGE_NAME='trainer_image'
TAG='latest'
TRAINER_IMAGE='gcr.io/{}/{}:{}'.format(PROJECT_ID, IMAGE_NAME, TAG)
```

```python
print(PROJECT_ID)

print(TAG)
```

return

'qwiklabs-gcp-00-1af1e75d90f7'

'latest'

### Build the trainer image

- Note: Please ignore any incompatibility ERROR that may appear for the packages visions as it will not affect the lab's functionality.

```python
!gcloud builds submit --timeout 15m --tag $TRAINER_IMAGE trainer_image
```

**Build the base image for Custom components**

```python
IMAGE_NAME='base_image'
TAG='latest'
BASE_IMAGE='gcr.io/{}/{}:{}'.format(PROJECT_ID, IMAGE_NAME, TAG)

!gcloud builds submit --timeout 15m --tag $BASE_IMAGE base_image
```

### Compile the Pipeline

- **KFP compiler** or **KFP SDK**를 이용해 DSL 컴파일 가능.

** (The pipeline can run using a security context of the GKE default node pool's service account or the service account defined in the user-gcp-sa secret of the Kubernetes namespace hosting KFP. If you want to use the user-gcp-sa service account you change the value of USE_KFP_SA to True.) **

대략 허가받은 account 만 pipeline 이용이 가능 하다는 뜻
→ 이용하고 싶으면 USE_KFP_SA를 True로 변환. 

```python
USE_KFP_SA = False

COMPONENT_URL_SEARCH_PREFIX = 'https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/'
RUNTIME_VERSION = '1.15'
PYTHON_VERSION = '3.7'

%env USE_KFP_SA={USE_KFP_SA}
%env BASE_IMAGE={BASE_IMAGE}
%env TRAINER_IMAGE={TRAINER_IMAGE}
%env COMPONENT_URL_SEARCH_PREFIX={COMPONENT_URL_SEARCH_PREFIX}
%env RUNTIME_VERSION={RUNTIME_VERSION}
%env PYTHON_VERSION={PYTHON_VERSION}
```

return 

env: USE_KFP_SA=False
env: BASE_IMAGE=gcr.io/qwiklabs-gcp-00-1af1e75d90f7/base_image:latest
env: TRAINER_IMAGE=gcr.io/qwiklabs-gcp-00-1af1e75d90f7/trainer_image:latest
env: COMPONENT_URL_SEARCH_PREFIX=https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/
env: RUNTIME_VERSION=1.15
env: PYTHON_VERSION=3.7

### Compile the pipeline

- Compile the `covertype_training_pipeline.py` with the `dsl-compile` command line

```python
# TO DO: Your code goes here

!dsl-compile --py pipeline/covertype_training_pipeline.py --output covertype_training_pipeline.yaml
```

return

`covertype_training_pipeline.yaml`

[covertype_training_pipeline.yaml](Training_2%202412c/covertype_%20709e8.md)

### 위에서 생성된 yaml 파일 결과 확인

```python
!head covertype_training_pipeline.yaml
```

return

"apiVersion": |-
  argoproj.io/v1alpha1
"kind": |-
  Workflow
"metadata":
  "annotations":
    "pipelines.kubeflow.org/pipeline_spec": |-
      {"description": "The pipeline training and deploying the Covertype classifierpipeline_yaml", "inputs": [{"name": "project_id"}, {"name": "region"}, {"name": "source_table_name"}, {"name": "gcs_root"}, {"name": "dataset_id"}, {"name": "evaluation_metric_name"}, {"name": "evaluation_metric_threshold"}, {"name": "model_id"}, {"name": "version_id"}, {"name": "replace_existing_version"}, {"default": "\n{\n    \"hyperparameters\":  {\n        \"goal\": \"MAXIMIZE\",\n        \"maxTrials\": 6,\n        \"maxParallelTrials\": 3,\n        \"hyperparameterMetricTag\": \"accuracy\",\n        \"enableTrialEarlyStopping\": True,\n        \"params\": [\n            {\n                \"parameterName\": \"max_iter\",\n                \"type\": \"DISCRETE\",\n                \"discreteValues\": [500, 1000]\n            },\n            {\n                \"parameterName\": \"alpha\",\n                \"type\": \"DOUBLE\",\n                \"minValue\": 0.0001,\n                \"maxValue\": 0.001,\n                \"scaleType\": \"UNIT_LINEAR_SCALE\"\n            }\n        ]\n    }\n}\n", "name": "hypertune_settings", "optional": true}, {"default": "US", "name": "dataset_location", "optional": true}], "name": "Covertype Classifier Training"}
  "generateName": |-
    covertype-classifier-training-

### Deploy the pipeline package

- 생성된 yaml 파일을 이용해 배포 진행

```python
PIPELINE_NAME='covertype_continuous_training'

# TO DO: Your code goes here

!kfp --endpoint $ENDPOINT pipeline upload \
-p $PIPELINE_NAME \
covertype_training_pipeline.yaml
```

return

Pipeline 4a16f02a-53c7-4f66-b163-b0904421f0a0 has been submitted

Pipeline Details
------------------
ID           4a16f02a-53c7-4f66-b163-b0904421f0a0
Name         covertype_continuous_training
Description
Uploaded at  2021-04-18T08:00:30+00:00
+-----------------------------+--------------------------------------------------+
| Parameter Name              | Default Value                                    |
+=============================+==================================================+
| project_id                  |                                                  |
+-----------------------------+--------------------------------------------------+
| region                      |                                                  |
+-----------------------------+--------------------------------------------------+
| source_table_name           |                                                  |
+-----------------------------+--------------------------------------------------+
| gcs_root                    |                                                  |
+-----------------------------+--------------------------------------------------+
| dataset_id                  |                                                  |
+-----------------------------+--------------------------------------------------+
| evaluation_metric_name      |                                                  |
+-----------------------------+--------------------------------------------------+
| evaluation_metric_threshold |                                                  |
+-----------------------------+--------------------------------------------------+
| model_id                    |                                                  |
+-----------------------------+--------------------------------------------------+
| version_id                  |                                                  |
+-----------------------------+--------------------------------------------------+
| replace_existing_version    |                                                  |
+-----------------------------+--------------------------------------------------+
| hypertune_settings          | {                                                |
|                             |     "hyperparameters":  {                        |
|                             |         "goal": "MAXIMIZE",                      |
|                             |         "maxTrials": 6,                          |
|                             |         "maxParallelTrials": 3,                  |
|                             |         "hyperparameterMetricTag": "accuracy",   |
|                             |         "enableTrialEarlyStopping": True,        |
|                             |         "params": [                              |
|                             |             {                                    |
|                             |                 "parameterName": "max_iter",     |
|                             |                 "type": "DISCRETE",              |
|                             |                 "discreteValues": [500, 1000]    |
|                             |             },                                   |
|                             |             {                                    |
|                             |                 "parameterName": "alpha",        |
|                             |                 "type": "DOUBLE",                |
|                             |                 "minValue": 0.0001,              |
|                             |                 "maxValue": 0.001,               |
|                             |                 "scaleType": "UNIT_LINEAR_SCALE" |
|                             |             }                                    |
|                             |         ]                                        |
|                             |     }                                            |
|                             | }                                                |
+-----------------------------+--------------------------------------------------+
| dataset_location            | US                                               |
+-----------------------------+--------------------------------------------------+

### Submitting pipeline runs

- You can trigger pipeline runs using an API from the KFP SDK or using KFP CLI. To submit the run using KFP CLI, execute the following commands. Notice how the pipeline's parameters are passed to the pipeline run.

### Check pipeline list

```python
!kfp --endpoint $ENDPOINT pipeline list
```

return

![Training_2%202412c/Untitled.png](Training_2%202412c/Untitled.png)

### Submit a run

→ 위에서 `covertype_continuous_training` ID를 가져와 변수 할당→ 위에서 covertype_continuous_training ID를 가져와 변수 할당

```python
# TO DO: REPLACE WITH YOUR PIPELINE ID
# 위의 covertype_continuous_training ID 를 가져옴
PIPELINE_ID='4a16f02a-53c7-4f66-b163-b0904421f0a0'

EXPERIMENT_NAME = 'Covertype_Classifier_Training'
RUN_ID = 'Run_001'
SOURCE_TABLE = 'covertype_dataset.covertype'
DATASET_ID = 'splits'
EVALUATION_METRIC = 'accuracy'
EVALUATION_METRIC_THRESHOLD = '0.69'
MODEL_ID = 'covertype_classifier'
VERSION_ID = 'v01'
REPLACE_EXISTING_VERSION = 'True'

GCS_STAGING_PATH = '{}/staging'.format(ARTIFACT_STORE_URI)
```

### Pipeline Run from KFP command

- EXPERIMENT_NAME is set to the experiment used to run the pipeline. You can choose any name you want. If the experiment does not exist it will be created by the command
- RUN_ID is the name of the run. You can use an arbitrary name
- PIPELINE_ID is the id of your pipeline. Use the value retrieved by the `kfp pipeline list` command
- GCS_STAGING_PATH is the URI to the Cloud Storage location used by the pipeline to store intermediate files. By default, it is set to the `staging` folder in your artifact store.
- REGION is a compute region for AI Platform Training and Prediction.

```python
# TO DO: Your code goes here
!kfp --endpoint $ENDPOINT run submit \
-e $EXPERIMENT_NAME \
-r $RUN_ID \
-p $PIPELINE_ID \
project_id=$PROJECT_ID \
gcs_root=$GCS_STAGING_PATH \
region=$REGION \
source_table_name=$SOURCE_TABLE \
dataset_id=$DATASET_ID \
evaluation_metric_name=$EVALUATION_METRIC \
evaluation_metric_threshold=$EVALUATION_METRIC_THRESHOLD \
model_id=$MODEL_ID \
version_id=$VERSION_ID \
replace_existing_version=$REPLACE_EXISTING_VERSION
```

return

![Training_2%202412c/Untitled%201.png](Training_2%202412c/Untitled%201.png)

### Monitoring the run

- KFP UI 를 통해 모니터링 가능
→ 이전 Endpoint에서 얻은 URL 을 통해 접근 가능
- https://[ENDPOINT]
    
    → [https://7d74ffdf8b783b10-dot-us-central1.pipelines.googleusercontent.com](https://7d74ffdf8b783b10-dot-us-central1.pipelines.googleusercontent.com/)
    
    - Endpoint
        
        ![Training_2%202412c/Endpoint%201.png](Training_2%202412c/Endpoint%201.png)
        
    - Monitoring
        
        ![Training_2%202412c/URL_Monitoring.png](Training_2%202412c/URL_Monitoring.png)