# Training_1 (caip-containers)

목표

1. 훈련 용 도커 이미지를 작성하는 방법
2. 하이퍼 파라미터 튜닝 학습 및 AI 플랫폼에 배포

### Installing Library

```python
import json
import os
import numpy as np
import pandas as pd
import pickle
import uuid
import time
import tempfile

from googleapiclient import discovery
from googleapiclient import errors

from google.cloud import bigquery
from jinja2 import Template
from kfp.components import func_to_container_op
from typing import NamedTuple

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
```

---

### AI Platform Pipeline - cluster 생성

![Training_1%2079a9b/Untitled.png](Training_1%2079a9b/Untitled.png)

![Training_1%2079a9b/Untitled%201.png](Training_1%2079a9b/Untitled%201.png)

![Training_1%2079a9b/Untitled%202.png](Training_1%2079a9b/Untitled%202.png)

![Training_1%2079a9b/Untitled%203.png](Training_1%2079a9b/Untitled%203.png)

![Training_1%2079a9b/Untitled%204.png](Training_1%2079a9b/Untitled%204.png)

---

### Prepare lab dataset

**환경변수 셋팅**

ㄴ> 데이터를 가져오기 위한 ID와 스키마 변수를 지정해 준다.

```python
PROJECT_ID=!(gcloud config get-value core/project)
PROJECT_ID=PROJECT_ID[0]
DATASET_ID='covertype_dataset'
DATASET_LOCATION='US'
TABLE_ID='covertype'
DATA_SOURCE='gs://workshop-datasets/covertype/small/dataset.csv'
SCHEMA='Elevation:INTEGER,Aspect:INTEGER,Slope:INTEGER,Horizontal_Distance_To_Hydrology:INTEGER,Vertical_Distance_To_Hydrology:INTEGER,Horizontal_Distance_To_Roadways:INTEGER,Hillshade_9am:INTEGER,Hillshade_Noon:INTEGER,Hillshade_3pm:INTEGER,Horizontal_Distance_To_Fire_Points:INTEGER,Wilderness_Area:STRING,Soil_Type:STRING,Cover_Type:INTEGER'
```

**BigQuery 이용해서 데이터 셋 생성**

ㄴ> 위에서 지정한 변수들을 이용해 데이터 셋을 생성

```python
!bq --location=$DATASET_LOCATION --project_id=$PROJECT_ID mk --dataset $DATASET_ID
```

```python
!bq --project_id=$PROJECT_ID --dataset_id=$DATASET_ID load \
--source_format=CSV \
--skip_leading_rows=1 \
--replace \
$TABLE_ID \
$DATA_SOURCE \
$SCHEMA
```

---

- 생략
    
    
    **Cloud Storage buckets list 추출**
    
    ```python
    !gsutil ls
    
    ### 총 3개의 결과 출력
    # gs://artifacts.qwiklabs-gcp-00-d59644d39516.appspot.com/
    # gs://qwiklabs-gcp-00-d59644d39516-kubeflowpipelines-default/
    # gs://qwiklabs-gcp-00-d59644d39516_cloudbuild/
    	
    # 과정에서는 kubeflowpipelines-default 를 이용
    ```
    
    ![Training_1%2079a9b/Untitled%205.png](Training_1%2079a9b/Untitled%205.png)
    
    - REGION - the compute region for AI Platform Training and Predict
    즉, 학습 과정이 진행되는 곳 (위 pipeline platform cluster 생성할 때 설정한 장소)
    - ARTIFACT_STORE - the Cloud Storage bucket created during installation of AI Platform Pipelines.
    즉, Platform 설치할 때 생성된 버켓
    
    ```python
    REGION = 'us-central1'
    ARTIFACT_STORE = 'gs://qwiklabs-gcp-00-d59644d39516-kubeflowpipelines-default' # TO DO: REPLACE WITH YOUR ARTIFACT_STORE NAME
    
    PROJECT_ID = !(gcloud config get-value core/project)
    PROJECT_ID = PROJECT_ID[0]
    DATA_ROOT='{}/data'.format(ARTIFACT_STORE) 
    JOB_DIR_ROOT='{}/jobs'.format(ARTIFACT_STORE)
    TRAINING_FILE_PATH='{}/{}/{}'.format(DATA_ROOT, 'training', 'dataset.csv')
    VALIDATION_FILE_PATH='{}/{}/{}'.format(DATA_ROOT, 'validation', 'dataset.csv')
    
    # 버켓의 data 폴더에서 training, validation data set 링크를 가져온다.
    TRAINING_FILE_PATH == 'gs://qwiklabs-gcp-00-d59644d39516-kubeflowpipelines-default/data/training/dataset.csv'
    VALIDATION_FILE_PATH == 'gs://qwiklabs-gcp-00-d59644d39516-kubeflowpipelines-default/data/validation/dataset.csv'
    ```
    

### 데이터 확인

[명령줄 도구 참조 | BigQuery | Google Cloud](https://cloud.google.com/bigquery/docs/reference/bq-cli-reference?hl=ko#bq_extract)

BigQuery 이용해서 데이터셋 확인

```python
%%bigquery
SELECT *
FROM `covertype_dataset.covertype`
```

![Training_1%2079a9b/Untitled%206.png](Training_1%2079a9b/Untitled%206.png)

---

BigQuery 로 추출한 데이터셋 **covertype_dataset.training** 으로 보내기
→ query : 전체의 80% 만 추출한다.

→ -n 0 : 반환할 행 수

→ destination_table : 쿼리 결과가 이곳에 저장됨.

→ replace : 대상 테이블이 쿼리 결과로 덮어쓰기

→ use_legacy_sql : false로 설정하여 표준 SQL을 기본 쿼리 구문으로 지정(표준 SQL 쿼리가 실행)

```python
!bq query \
-n 0 \
--destination_table covertype_dataset.training \
--replace \
--use_legacy_sql=false \
'SELECT * \
FROM `covertype_dataset.covertype` AS cover \
WHERE \
MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(cover))), 10) IN (0, 1, 2, 3, 4, 5, 6, 7)'
```

**TRAINING_FILE_PATH로 Training_set을 CSV형식으로 생성**

→ bq extract 명령어는 --destination_format 플래그와 함께 사용

→ destination_format : 내보내는 데이터 형식
→ TRAINING_FILE_PATH : 저장할 파일 위치

```python
!bq extract \
--destination_format CSV \
covertype_dataset.training \
$TRAINING_FILE_PATH
```

**위 링크를 현재 폴더의 training.csv 라는 이름으로 카피**

→ 원래는 다른 방법이 있지만, cloud에서 import가 되지 않아 카피해 옴.

→ kwiklab 문제

```python
!gsutil cp $TRAINING_FILE_PATH ./training.csv
```

**training_set, validation_set 위치 재설정 및 shape check**

```python
TRAINING_FILE_PATH = './training.csv'
VALIDATION_FILE_PATH = './validtaion.csv'

df_train = pd.read_csv(TRAINING_FILE_PATH)
df_validation = pd.read_csv(VALIDATION_FILE_PATH)
print(df_train.shape)
### (80072, 13)

print(df_validation.shape)
### (9836, 13)
```

---

### **Develop a training application (하이퍼 튜닝 기능 이용하지 않는 경우 코드)**

```python
# 0~9번째 컬럼 numeric type
numeric_feature_indexes = slice(0, 10)

# 10~11번째 컬럼 categorical type
categorical_feature_indexes = slice(10, 12)

# 정규화처리 및 원-핫 인코딩
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_feature_indexes),
        ('cat', OneHotEncoder(), categorical_feature_indexes) 
    ])

# Pipeline에 전처리 및 SGD Classifier 적용
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SGDClassifier(loss='log', tol=1e-3))
])
```

**Numeric to float64**

→ numeric 형식의 컬럼들만 바꾸어준다.

```python
num_features_type_map = {feature: 'float64' for feature in df_train.columns[numeric_feature_indexes]}

df_train = df_train.astype(num_features_type_map)
df_validation = df_validation.astype(num_features_type_map)
```

**Run pipeline and Calculate model accuracy**
→ 여기서는 파라미터가 각각 1개씩만 지정되어있음.

```python
X_train = df_train.drop('Cover_Type', axis=1)
y_train = df_train['Cover_Type']
X_validation = df_validation.drop('Cover_Type', axis=1)
y_validation = df_validation['Cover_Type']

# 각 하나의 파라미터만 지정되어있다.
pipeline.set_params(classifier__alpha=0.001, classifier__max_iter=200)
pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_validation, y_validation)

print(accuracy)
### 0.7008151325
```

---

### Hyperparameter tuning application

training_app 폴더를 생성
- exist_ok : True 면 없는 경우 생성 있으면 패스

```python
TRAINING_APP_FOLDER = 'training_app'
os.makedirs(TRAINING_APP_FOLDER, exist_ok=True)
```

**training_app 폴더에 train.py 파일을 생성한다.**

```python
%%writefile {TRAINING_APP_FOLDER}/train.py

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys

import fire
import pickle
import numpy as np
import pandas as pd

import hypertune

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 위에서 진행된 것과 같은 내용
# 공통으로 진행되는 전처리 과정이 포함되어 있다.
def train_evaluate(job_dir, training_dataset_path, 
                   validation_dataset_path, alpha, max_iter, hptune):

    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)

    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    numeric_feature_indexes = slice(0, 10)
    categorical_feature_indexes = slice(10, 12)

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_feature_indexes),
        ('cat', OneHotEncoder(), categorical_feature_indexes) 
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SGDClassifier(loss='log',tol=1e-3))
    ])

    num_features_type_map = {feature: 'float64' for feature 
                             in df_train.columns[numeric_feature_indexes]}
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map) 

    print('Starting training: alpha={}, max_iter={}'.format(alpha, max_iter))
    X_train = df_train.drop('Cover_Type', axis=1)
    y_train = df_train['Cover_Type']

    pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
    pipeline.fit(X_train, y_train)

# 하이퍼 튜닝 부분
# 
    if hptune:
    # TO DO: Your code goes here to score the model with the validation data and capture the result
    # with the hypertune library
        X_validation = df_validation.drop('Cover_Type', axis=1)
        y_validation = df_validation['Cover_Type']
        accuracy = pipeline.score(X_validation, y_validation)
        print('Model accuracy: {}'.format(accuracy))
        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
          hyperparameter_metric_tag='accuracy',
          metric_value=accuracy
        )

    # Save the model
    if not hptune:
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipeline, model_file)
        gcs_model_path = "{}/{}".format(job_dir, model_filename)
        subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
                          stderr=sys.stdout)
        print("Saved model in: {}".format(gcs_model_path))

if __name__ == "__main__":
    fire.Fire(train_evaluate)
```

**Docker 파일 생성 (training_app 폴더에 생성됨)**

→ '%%writefile {TRAINING_APP_FOLDER}/Dockerfile' 명령어를 이용해서 문구 저장

→ train.py [](http://train.py)파일을 app 폴더로 복사

→ 필요한 라이브러리를 다운받을 수 있도록한다.

```python
%%writefile {TRAINING_APP_FOLDER}/Dockerfile

FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire cloudml-hypertune scikit-learn==0.20.4 pandas==0.24.2
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
```

**위 도커 이미지 빌드**

```python
IMAGE_NAME='trainer_image'
IMAGE_TAG='latest'
IMAGE_URI='gcr.io/{}/{}:{}'.format(PROJECT_ID, IMAGE_NAME, IMAGE_TAG)
```

### Submit an AI Platform hyperparameter tuning job (yaml 파일 생성)

hyperparameter configuration file 생성 (training_app 폴더에 생성된다.)

→ '%%writefile {TRAINING_APP_FOLDER}/hptuning_config.yaml' 을 이용해서 아래 문구 저장

→ 원하는 하이퍼 파라미터 내역을 적어준다.

→ max_iter : 200, 300

→ alpha : 0.00001, 0.001

```python
%%writefile {TRAINING_APP_FOLDER}/hptuning_config.yaml

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

trainingInput:
    hyperparameters:
        goal: MAXIMIZE
        maxTrials: 4
        maxParallelTrials: 4
        hyperparameterMetricTag: accuracy
        enableTrialEarlyStopping: TRUE 
        params:
            # TO DO: Your code goes here
### 여기에 원하는 파라미터를 넣어준다.
        - parameterName: max_iter
          type: DISCRETE
          discreteValues: [
              200,
              500
          ]
        - parameterName: alpha
          type: DOUBLE
          minValue: 0.00001
          maxValue: 0.001
          scaleType: UNIT_LINEAR_SCALE
```

**Hyperparameter tuning job 시작**

→ yaml 파일

```python
JOB_NAME = "JOB_{}".format(time.strftime("%Y%m%d_%H%M%S"))
JOB_DIR = "{}/{}".format(JOB_DIR_ROOT, JOB_NAME)
SCALE_TIER = "BASIC"

# Todo hptuning
!gcloud ai-platform jobs submit training $JOB_NAME \
--region=$REGION \ ## us-central1
--job-dir=$JOB_DIR \ ## gs://qwiklabs-gcp-00-d59644d39516-kubeflowpipelines-default/jobs/JOB_20210414_044544
--master-image-uri=$IMAGE_URI \ ## 위에서 빌드한 이미지 gcr.io/qwiklabs-gcp-00-d59644d39516/trainer_image:latest
--scale-tier=$SCALE_TIER \ ## BASIC
--config $TRAINING_APP_FOLDER/hptuning_config.yaml \ ## 위에서 작성한 yaml 파일 (하이퍼파라미터 튜닝 정보가 들어있다)
-- \
--training_dataset_path=$TRAINING_FILE_PATH \ ## training data 위치
--validation_dataset_path=$VALIDATION_FILE_PATH \
--hptune ## hptune 존재 
```

job_dir, 

training_dataset_path, 

validation_dataset_path, 

alpha, (hptune에 들어있음) 

max_iter, (hptune에 들어있음)

hptune

return

![Training_1%2079a9b/Untitled%207.png](Training_1%2079a9b/Untitled%207.png)

**Monitoring**

```python
!gcloud ai-platform jobs describe $JOB_NAME
```

![Training_1%2079a9b/Untitled%208.png](Training_1%2079a9b/Untitled%208.png)

**Stream-Log data**

```python
!gcloud ai-platform jobs stream-logs $JOB_NAME
```

![Training_1%2079a9b/Untitled%209.png](Training_1%2079a9b/Untitled%209.png)

**Getting HP-tuning Result**

```python
ml = discovery.build('ml', 'v1')

job_id = 'projects/{}/jobs/{}'.format(PROJECT_ID, JOB_NAME)
request = ml.projects().jobs().get(name=job_id)

try:
    response = request.execute()
except errors.HttpError as err:
    print(err)
except:
    print("Unexpected error")
```

```python
response
```

![Training_1%2079a9b/Untitled%2010.png](Training_1%2079a9b/Untitled%2010.png)

**1번째 데이터 뽑기**

```python
response['trainingOutput']['trials'][0]
```

```python
'trailID' : '2', 
'hyperparameters' : {'alpha' : '0.000505', 'max_iter' :'500'}.
'finalmetrix' : {'trainingStep' : '1', 'objectiveValue' : '0.70201'},
'startTime' : '2021-04-14T04:26:24:26.13452',
'endTiime' : '2021-04-14T04:52:462',
'state' : 'SUCCEEDED'}
```

**Using best Hyperparameter**

→ Training script 'model.pkl'로 저장된다→ Training script 'model.pkl'로 저장된다.

```python
alpha = response['trainingOutput']['trials'][0]['hyperparameters']['alpha']
max_iter = response['trainingOutput']['trials'][0]['hyperparameters']['max_iter']

JOB_NAME = "JOB_{}".format(time.strftime("%Y%m%d_%H%M%S"))
JOB_DIR = "{}/{}".format(JOB_DIR_ROOT, JOB_NAME)
SCALE_TIER = "BASIC"

!gcloud ai-platform jobs submit training $JOB_NAME \
--region=$REGION \
--job-dir=$JOB_DIR \
--master-image-uri=$IMAGE_URI \
--scale-tier=$SCALE_TIER \
-- \
--training_dataset_path=$TRAINING_FILE_PATH \
--validation_dataset_path=$VALIDATION_FILE_PATH \
--alpha=$alpha \  ## 따로 지정해준다
--max_iter=$max_iter \ ## 따로 지정해준다
--nohptune  ## hptune을 이용하지 않는다.
```

![Training_1%2079a9b/Untitled%2011.png](Training_1%2079a9b/Untitled%2011.png)

**JOB_DIR 에 저장된 Pkl 파일 확인**

```python
!gsutil ls $JOB_DIR
```

return

`gs://qwiklabs-gcp-00-d59644d39516-kubeflowpipelines-default/jobs/JOB_20210414_050251/model.pkl`

### Deploy the model to AI Platform Prediction

Create model

```python
model_name = 'forest_cover_classifier'
labels = "task=classifier,domain=forestry"

# TO DO: You code goes here
!gcloud ai-platform models create  $model_name \
 --regions=$REGION \
 --labels=$labels
```

![Training_1%2079a9b/Untitled%2012.png](Training_1%2079a9b/Untitled%2012.png)

**Create model version**

```python
model_version = 'v01'
# TO DO: Complete the command

!gcloud ai-platform versions create {model_version} \
 --model={model_name} \
 --origin=$JOB_DIR \
 --runtime-version=1.15 \
 --framework=scikit-learn \
 --python-version=3.7\
 --region global
```

![Training_1%2079a9b/Untitled%2013.png](Training_1%2079a9b/Untitled%2013.png)

**Serve Predictions**

```python
input_file = 'serving_instances.json'

with open(input_file, 'w') as f:
    for index, row in X_validation[:10].iterrows():
        f.write(json.dumps(list(row.values)))
        f.write('\n')
```

```python
!cat $input_file
```

![Training_1%2079a9b/Untitled%2014.png](Training_1%2079a9b/Untitled%2014.png)

**Invoke the model**

```python
# TO DO: Complete the command
!gcloud ai-platform predict \
--model $model_name \
--version $model_version \
--json-instances $input_file\
--region global
```

![Training_1%2079a9b/Untitled%2015.png](Training_1%2079a9b/Untitled%2015.png)