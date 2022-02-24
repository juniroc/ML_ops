# Vertex AI

### Tabular data AutoML 돌려보기

### GCP 에서 Vertex AI

- 90일 300$ 무료 크레딧 받아 접속

### 새 프로젝트 생성

![Untitled](Vertex%20AI%200b9a4/Untitled.png)

### API 사용 설정

![Untitled](Vertex%20AI%200b9a4/Untitled%201.png)

### Vertex AI 선택

![Untitled](Vertex%20AI%200b9a4/Untitled%202.png)

- GCP 목록 중 Vertex AI 선택

### 대쉬보드

![Untitled](Vertex%20AI%200b9a4/Untitled%203.png)

### Dataset 선택

![Untitled](Vertex%20AI%200b9a4/Untitled%204.png)

- `tabular` 데이터 선택
- 리전은 `default` 로 선택

### dataset 추가

![Untitled](Vertex%20AI%200b9a4/Untitled%205.png)

- `gs://cloud-ml-tables-data/bank-marketing.csv` 경로 추가

---

### 데이터 통계 추출

- 오른쪽에 `통계 생성` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%206.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%207.png)

- 위와 같이 컬럼별로 `누락률(개수)` `고윳값`을 알려줌

### 모델학습 실행

![Untitled](Vertex%20AI%200b9a4/Untitled%208.png)

- `새 모델 학습` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%209.png)

- `계속` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2010.png)

- `Target column` 을 `Deposit` 으로 설정
- `2` = yes, `1` = no
- `계속` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2011.png)

- `컬럼 이름` 클릭할 경우 아래와 같이 **컬럼 명세 출력**

![Untitled](Vertex%20AI%200b9a4/Untitled%2012.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2013.png)

ex) Deposit 내용

![Untitled](Vertex%20AI%200b9a4/Untitled%2014.png)

- `계속` 클릭

### 학습에 사용할 최대 노드 시간 입력

![Untitled](Vertex%20AI%200b9a4/Untitled%2015.png)

- `1` 로 입력
- `학습 시작` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2016.png)

- 위와 같이 `snack bar` 생성

`학습` 을 누르면 다음과같이 학습에 관련된 정보 출력됨

![Untitled](Vertex%20AI%200b9a4/Untitled%2017.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2018.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2019.png)

### 학습 완료 확인

- 메일로 `학습완료`를 알림

![Untitled](Vertex%20AI%200b9a4/Untitled%2020.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2021.png)

---

### After Training

![Untitled](Vertex%20AI%200b9a4/Untitled%2022.png)

- `모델`을 들어가면 완성된 모델이 나옴

### 모델 Validation

![Untitled](Vertex%20AI%200b9a4/Untitled%2023.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2024.png)

---

### Deployment & Test

![Untitled](Vertex%20AI%200b9a4/Untitled%2025.png)

- `엔드포인트에 배포` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2026.png)

- `Endpoint` 이름 설정
- `Endpoint` 생성

![Untitled](Vertex%20AI%200b9a4/Untitled%2027.png)

- `최소 컴퓨팅 노드 수` : `1`
- `machine type` : `n1-standard-2` (가장 저렴한 것으로 설정)

![Untitled](Vertex%20AI%200b9a4/Untitled%2028.png)

- `완료` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2029.png)

- `배포` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2030.png)

- 약 `10~15분` (또는 그이상..) 정도 소요
- 이것 또한 이메일로 완료되면 알림

![Untitled](Vertex%20AI%200b9a4/Untitled%2031.png)

### 엔드포인트

![Untitled](Vertex%20AI%200b9a4/Untitled%2032.png)

- `엔드포인트` 항목에 추가된 것도 확인 가능

### 배포

- 배포가 완료되면 아래와 같이 Inference test 가 활성됨

![Untitled](Vertex%20AI%200b9a4/Untitled%2033.png)

- `예측` 누르면 다음과 같이 오른쪽에 생성됨

![Untitled](Vertex%20AI%200b9a4/Untitled%2034.png)

### REST 나 PYTHON을 통해 요청 가능

![Untitled](Vertex%20AI%200b9a4/Untitled%2035.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2036.png)

- `ENDPOINT_ID` , `PROJECT_ID` , `INPUT_DATA_FILE` , `LOCATION_ID` 등 환경변수 맞춰주어 인퍼런스 요청하면 됨

### Remove model Deployment

![Untitled](Vertex%20AI%200b9a4/Untitled%2037.png)

- `endpoint` 탭으로 가서 박스 체크 후 제거해주면 간단히 배포된 모델을 제거할 수 있음

---

## Notebook 생성

![Untitled](Vertex%20AI%200b9a4/Untitled%2038.png)

- `운영체제(OS)` , `env(framework)` 설정
- 가장 저렴한 `n1-standard-1` 으로 생성

![Untitled](Vertex%20AI%200b9a4/Untitled%2039.png)

- 디스크는 `최소` `100GB` `이상`이어야함
- `생성` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2040.png)

- 약간의 시간 소요.. (인스턴스 생성시간)

![Untitled](Vertex%20AI%200b9a4/Untitled%2041.png)

- 완료되면 아래와 같이 생성됨

![Untitled](Vertex%20AI%200b9a4/Untitled%2042.png)

- `JUPYTERLAB 열기` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2043.png)

- `Notebook` 의 `Python3` 클릭

![Untitled](Vertex%20AI%200b9a4/Untitled%2044.png)

- 이렇게 쓸수 있음

![Untitled](Vertex%20AI%200b9a4/Untitled%2045.png)

- `Local` 에 있는 파일도 드래그하여 다음과 같이 넣어 줄 수 있음.

![Untitled](Vertex%20AI%200b9a4/Untitled%2046.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2047.png)

- 노트북 제거할 때는 창 위에 있는 `삭제` 버튼을 누르면 됨
- 또는 이용하지 않을 경우 `중지`

![Untitled](Vertex%20AI%200b9a4/Untitled%2048.png)

### Vertex AI Dashboard

![Untitled](Vertex%20AI%200b9a4/Untitled%2049.png)

---

### API 이용한 인퍼런스

```bash
# [START aiplatform_predict_tabular_classification_sample]
from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

# [END aiplatform_predict_tabular_classification_sample]
```

```bash
instances_={
    "Age" : "39.0",
    "Job" : "blue-collar",
    "MaritalStatus" : "married",
    "Education" : "secondary",
    "Default" : "no",
    "Balance" : "449.0",
    "Housing" : "yes",
    "Loan" : "no",
    "Contact" : "cellular",
    "Day" : "16.0",
    "Month" : "may",
    "Duration" : "180.0",
    "Campaign" : "2.0",
    "PDays" : "-1.0",
    "Previous" : "0.0",
    "POutcome" : "unknown"}
```

```bash
predict_tabular_classification_sample(
    project="1099081172311",
    endpoint_id="7840001691159101440",
    location="us-central1",
    instance_dict=instances_
)
```

![Untitled](Vertex%20AI%200b9a4/Untitled%2050.png)

---

## Remote Inference

### Credentials

[Getting started with authentication | Authentication | Google Cloud](https://cloud.google.com/docs/authentication/getting-started)

![Untitled](Vertex%20AI%200b9a4/Untitled%2051.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2052.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2053.png)

- 서비스 계정 생성됨을 알 수 있음

- `mj_lee` 로 들어가서

![Untitled](Vertex%20AI%200b9a4/Untitled%2054.png)

- 키 항목에서 `키 추가` 눌러서 `새 키 만들기`

![Untitled](Vertex%20AI%200b9a4/Untitled%2055.png)

- 위와 같이 생성됨
- 추가적으로 해당 키는 `json` 형태로 `local` 에 다운로드 됨

![Untitled](Vertex%20AI%200b9a4/Untitled%2056.png)

- 생성된 파일을 `~/.zshrc` 에 추가하면 됨
- `vi ~/.zshrc` 에 해당 파일을 `GOOGLE_APPLICATION_CREDENTIALS` 로 추가해주면 됨

`~/.zshrc`

```bash
## 맨아래로 내려가서 

export GOOGLE_APPLICATION_CREDENTIALS="/Users/minjunlee/Downloads/.com.google.Chrome.TxcUN4"
```

![Untitled](Vertex%20AI%200b9a4/Untitled%2057.png)

`inf_.py` in local

```python
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/1099081172311/locations/us-central1/endpoints/7840001691159101440"
)

aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project='1099081172311',

    # the Vertex AI region you will use
    # defaults to us-central1
    location='us-central1',

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://cloud-ai-platform-cc12072c-b392-4851-9411-19d3fb5302ca',

    # custom google.auth.credentials.Credentials
    # environment default creds used if not set
    # credentials=" ",

    # customer managed encryption key resource name
    # will be applied to all Vertex AI resources if set
#     encryption_spec_key_name=my_encryption_key_name,

    # the name of the experiment to use to track
    # logged metrics and parameters
    experiment='my-experiment',

    # description of the experiment above
    experiment_description='my experiment decsription'
)

# [START aiplatform_predict_tabular_classification_sample]
from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

# [END aiplatform_predict_tabular_classification_sample]

instances_={
    "Age" : "39.0",
    "Job" : "blue-collar",
    "MaritalStatus" : "married",
    "Education" : "secondary",
    "Default" : "no",
    "Balance" : "449.0",
    "Housing" : "yes",
    "Loan" : "no",
    "Contact" : "cellular",
    "Day" : "16.0",
    "Month" : "may",
    "Duration" : "180.0",
    "Campaign" : "2.0",
    "PDays" : "-1.0",
    "Previous" : "0.0",
    "POutcome" : "unknown"}

predict_tabular_classification_sample(
    project="1099081172311",
    endpoint_id="7840001691159101440",
    location="us-central1",
    instance_dict=instances_
)
```

![Untitled](Vertex%20AI%200b9a4/Untitled%2058.png)

---

---

## Model Download

![Untitled](Vertex%20AI%200b9a4/Untitled%2059.png)

- 다운 받을 모델 폴더의 코드 복사

[Install gsutil | Cloud Storage | Google Cloud](https://cloud.google.com/storage/docs/gsutil_install#mac)

```bash
## 위 페이지에 접속하여 해당 OS 별 패키지 다운로드 후 인스톨

## google auth 이용해 접속 로그인
gcloud auth login

## gsutil -m cp -r \
  "gs://cloud-ai-platform-cc12072c-b392-4851-9411-19d3fb5302ca/model-7720898193591894016/tf-saved-model/" \
  .
```

![Untitled](Vertex%20AI%200b9a4/Untitled%2060.png)

- 위와 같이 `tf-saved-model` 이 다운로드됨

### Download 된 모델 도커 컨테이너로 띄우기

```bash
sudo docker run -v /mnt/mj_lee/vertex_ai_test/model-7720898193591894016/tf-saved-model/models/:/models/default -p 8093:8080 -it us-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:20211210_1325_RC00
```

- `-v` : 모델 저장된 위치 지정
- `-p` : 포트
- `image` :  아래 사진의 container_uri 에 적힌 것 이용

![Untitled](Vertex%20AI%200b9a4/Untitled%2061.png)

### Inference

![Untitled](Vertex%20AI%200b9a4/Untitled%2062.png)

```json
{"instances":[{"Age" : "39.0",
             "Job" : "blue-collar",
             "MaritalStatus" : "married", 
             "Education" : "secondary", 
             "Default" : "no", 
             "Balance" : "449.0", 
             "Housing" : "yes", 
             "Loan" : "no", 
             "Contact" : "cellular", 
             "Day" : "16.0", 
             "Month" : "may", 
             "Duration" : "180.0", 
             "Campaign" : "2.0", 
             "PDays" : "-1.0", 
             "Previous" : "0.0", 
             "POutcome" : "unknown"}]}
```

- 위와 같이 `json` 파일 작성한 후

`curl -X POST --data @./test.json http://localhost:8093/predict`

- 위 REST API 명령어를 통해 요청

![Untitled](Vertex%20AI%200b9a4/Untitled%2063.png)

---

### Custom training

![Untitled](Vertex%20AI%200b9a4/Untitled%2064.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2065.png)

![Untitled](Vertex%20AI%200b9a4/Untitled%2066.png)

- 패키지 위치
: `cloud-samples-data/ai-platform/hello-custom/hello-custom-sample-v1.tar.gz`
- Python 모듈
: `trainer.task` 
→ 이건 `trainer/task.py` 라고 생각하면됨

---

![Untitled](Vertex%20AI%200b9a4/Untitled%2067.png)

---

### 도커 파일 내부

![Untitled](Vertex%20AI%200b9a4/Untitled%2068.png)

- 위와같이 `model.py` `task.py` 를 넣고 실행해주는 것

- 하지만 실제로 내부 소스코드를 어떤식으로 연결해주어야하는지는 자세히 나와있지 않아서 러닝커브가 높음

---

### 학습 이후

- `Endpoint` 로 배포

![Untitled](Vertex%20AI%200b9a4/Untitled%2069.png)

`trainer/task.py`

```python
import logging
import os

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_WIDTH = 128

def normalize_img(image):
    """Normalizes image.

    * Resizes image to IMG_WIDTH x IMG_WIDTH pixels
    * Casts values from `uint8` to `float32`
    * Scales values from [0, 255] to [0, 1]

    Returns:
      A tensor with shape (IMG_WIDTH, IMG_WIDTH, 3). (3 color channels)
    """
    image = tf.image.resize_with_pad(image, IMG_WIDTH, IMG_WIDTH)
    return image / 255.

def normalize_img_and_label(image, label):
    """Normalizes image and label.

    * Performs normalize_img on image
    * Passes through label unchanged

    Returns:
      Tuple (image, label) where
      * image is a tensor with shape (IMG_WIDTH, IMG_WIDTH, 3). (3 color
        channels)
      * label is an unchanged integer [0, 4] representing flower type
    """
    return normalize_img(image), label

if 'AIP_MODEL_DIR' not in os.environ:
    raise KeyError(
        'The `AIP_MODEL_DIR` environment variable has not been' +
        'set. See https://cloud.google.com/ai-platform-unified/docs/tutorials/image-recognition-custom/training'
    )
output_directory = os.environ['AIP_MODEL_DIR']

logging.info('Loading and preprocessing data ...')
dataset = tfds.load('tf_flowers:3.*.*',
                    split='train',
                    try_gcs=True,
                    shuffle_files=True,
                    as_supervised=True)
dataset = dataset.map(normalize_img_and_label,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
dataset = dataset.shuffle(1000)
dataset = dataset.batch(128)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

logging.info('Creating and training model ...')
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,
                           3,
                           padding='same',
                           activation='relu',
                           input_shape=(IMG_WIDTH, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(5)  # 5 classes
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(dataset, epochs=10)

logging.info(f'Exporting SavedModel to: {output_directory}')
# Add softmax layer for intepretability
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model.save(output_directory)
```

`function/main.py`

```python
import logging
from operator import itemgetter
import os

from flask import jsonify
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import requests
import tensorflow as tf

IMG_WIDTH = 128
COLUMNS = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

aip_client = aiplatform.gapic.PredictionServiceClient(client_options={
    'api_endpoint': 'us-central1-aiplatform.googleapis.com'
})
aip_endpoint_name = f'projects/{os.environ["GCP_PROJECT"]}/locations/us-central1/endpoints/{os.environ["ENDPOINT_ID"]}'

def get_prediction(instance):
    logging.info('Sending prediction request to AI Platform ...')
    try:
        pb_instance = json_format.ParseDict(instance, Value())
        response = aip_client.predict(endpoint=aip_endpoint_name,
                                      instances=[pb_instance])
        return list(response.predictions[0])
    except Exception as err:
        logging.error(f'Prediction request failed: {type(err)}: {err}')
        return None

def preprocess_image(image_url):
    logging.info(f'Fetching image from URL: {image_url}')
    try:
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        assert image_response.headers.get('Content-Type') == 'image/jpeg'
    except (ConnectionError, requests.exceptions.RequestException,
            AssertionError):
        logging.error(f'Error fetching image from URL: {image_url}')
        return None

    logging.info('Decoding and preprocessing image ...')
    image = tf.io.decode_jpeg(image_response.content, channels=3)
    image = tf.image.resize_with_pad(image, IMG_WIDTH, IMG_WIDTH)
    image = image / 255.
    return image.numpy().tolist()  # Make it JSON-serializable

def classify_flower(request):
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows POST requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Disallow non-POSTs
    if request.method != 'POST':
        return ('Not found', 404)

    # Set CORS headers for the main request
    headers = {'Access-Control-Allow-Origin': '*'}

    request_json = request.get_json(silent=True)
    if not request_json or not 'image_url' in request_json:
        return ('Invalid request', 400, headers)

    instance = preprocess_image(request_json['image_url'])
    if not instance:
        return ('Invalid request', 400, headers)

    raw_prediction = get_prediction(instance)
    if not raw_prediction:
        return ('Error getting prediction', 500, headers)

    probabilities = zip(COLUMNS, raw_prediction)
    sorted_probabilities = sorted(probabilities,
                                  key=itemgetter(1),
                                  reverse=True)
    return (jsonify(sorted_probabilities), 200, headers)
```