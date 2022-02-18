# Part 1. 프로덕션 레벨에서 필요한 기능 테스트

[https://engineering.linecorp.com/ko/blog/mlops-bentoml-1/#0](https://engineering.linecorp.com/ko/blog/mlops-bentoml-1/#0)

[](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/tree/main/feast_nni_bentoml_db_)

### MLflow 와 연동하여 BentoML 이용

- MLflow 는 주로 **실험 관리, 모델러 간 커뮤니케이션 용도** **및** **모델 저장소**로 사용

```python
# MLflow client 생성
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri=mlflow_endpoint)

# 모델 이름 정의
model_name="{model_name}"

# 모델 repository 검색 및 조회
filter_string = "name='{}'".format(model_name)
results = client.search_model_versions(filter_string)
for res in results:
    print("name={}; run_id={}; version={}; current_stage={}".format(res.name, res.run_id, res.version, res.current_stage))

# Production stage 모델 버전 선택
for res in results:
    if res.current_stage == "Production":
        deploy_version = res.version

# MLflow production 모델 버전 다운로드 URI 획득
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
model_uri = client.get_model_version_download_uri(model_name, deploy_version)

# 모델 다운로드
download_path = "{local_download_path}"
mlflow_run_id = "{run_id}"
mlflow_run_id_artifacts_name = "{artifacts_model_name}"
client.download_artifacts(mlflow_run_id, mlflow_run_id_artifacts_name, dst_path=download_path)

# 다운로드 모델 load & predict 예시
reconstructed_model = mlflow.{framework}.load_model("{download_path}/{model_name}".format(download_path=download_path,model_name=mlflow_run_id_artifacts_name))
output = reconstructed_model.predict(input_feature)
```

### BentoML의 멀티 모델 기능

- sklearn, keras, pytorch, FastAI, XGB, LGB, TF + custom model 등 라이브러리에서 나온 모델을 동시에 연동 가능

```python
import xgboost as xgb
import pandas as pd
import torch
import logging
from bentoml import BentoService, api, artifacts, env
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.frameworks.xgboost import XgboostModelArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import JsonInput, JsonOutput
from model_api_input_validator import ModelApiInputValidator

@env(infer_pip_packages=True)
@artifacts([ # Framework별 모델 이름 지정
    KerasModelArtifact('keras_model'),
    XgboostModelArtifact('xgb_model'),
    PytorchModelArtifact('pytorch_model')
])
class ModelApiService(BentoService):
        
    @api(input=ModelApiInputValidator())
    def predict(self, df: pd.DataFrame):
        # API 인증 처리
        
        # Input Feature 전처리
        
        # keras_model 예측
        keras_output = self.artifacts.keras_model.predict(df)
        
        # xgb_model 예측 
        xgb_output = self.artifacts.xgb_model.predict(xgb.DMatrix(df))
        # pytorch_model 예측
        pytorch_output = self.artifacts.pytorch_model(torch.from_numpy(df))
        
        # 예측 결과 후처리
       
        # Logging
        
        # 결과 Return

				return result
```

### 커스텀 URL 기능

```python
@env(infer_pip_packages=True)
@artifacts([
   ...
])
class ModelApiService(BentoService):
        
    @api(input=ModelApiInputValidator(),
         output=JsonOutput(),
         route="v1/service_name/predict",
         mb_max_latency=200,
         mb_max_batch_size=500,
         batch=False
        )
    def predict_v1(self, df: pd.DataFrame):
       ...
        return result

    @api(input=ModelApiInputValidator(),
         output=JsonOutput(),
         route="v1/service_name/batch",
         batch=True
        )
    def predict_v1_batch(self, df: pd.DataFrame):
       ...
        return [result]

    @api(input=ModelApiInputValidator(),
         output=JsonOutput(),
         route="v2/service_name/predict",
         mb_max_latency=300,
         mb_max_batch_size=1000,
         batch=False
        )
    def predict_v2(self, df: pd.DataFrame):
       ...
        return result
```

- 버저닝 기능 적용시 route 파라미터를 사용하면 좋다고 생각했음

### 지정한 이름으로 API URL 생성가능

- Swagger를 통해 확인

![Untitled](Part%201%20%E1%84%91%E1%85%B3%E1%84%85%2079c6a/Untitled.png)

- 다음과 같이 **버전/모델/데이터특성** 별로 url 생성됨

### 커스텀 인풋 전처리 및 배치 지원 기능

- dataframe, json, tensor, 이미지, 문자열, 파일 등 다양한 유형의 인풋 지원
- 단 추가로 커스터마이징(인풋 처리) 또는 검증이 필요할 수 있음
→ 이 경우엔, 검증 과정에서 known error 로 처리해 응답 코드를 200으로 송신해 비즈니스 로직에서 재처리하는 경우도 있음

```python
import json
import traceback
import pandas as pd
from enum import Enum
from typing import Iterable, Sequence, Tuple
from bentoml.adapters.string_input import StringInput
from bentoml.types import InferenceTask, JsonSerializable

ApiFuncArgs = Tuple[
    Sequence[JsonSerializable],
]

# 사용자 정의 ERROR CODE
class ErrorCode(Enum):
    INPUT_FORMAT_INVALID = ("1000", "Missing df_data")
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg

# 사용자 정의 Exception Class
class MyCustomException(Exception):
    def __init__(self ,code, msg):
        self.code = code
        self.msg = msg

class MyCustomDataframeInput(StringInput):
    def extract_user_func_args(self, tasks: Iterable[InferenceTask[str]]) -> ApiFuncArgs:
        json_inputs = []
        # tasks 객체는 Inference로 들어온 요청
        for task in tasks:
            try:
                # task.data는 request body 데이터를 의미
                parsed_json = json.loads(task.data)
                # 예외 처리 예시
                if parsed_json.get("df_data") is None:
                    raise MyCustomException(
                        msg=ErrorCode.INPUT_FORMAT_INVALID.msg, code=ErrorCode.INPUT_FORMAT_INVALID.code
                    )
                else:
                    # batch 처리를 위한 부분
                    df_data = parsed_json.get("df_data")
                    task.batch = len(df_data)
                    json_inputs.extend(df_data)
            except json.JSONDecodeError:
                task.discard(http_status=400, err_msg="Not a valid JSON format")
            except MyCustomException as e:
                task.discard(http_status=200, err_msg="Msg : {msg}, Error Code : {code}".format(msg=e.msg, code=e.code))
            except Exception:
                err = traceback.format_exc()
                task.discard(http_status=500, err_msg=f"Internal Server Error: {err}")

        # Dataframe 변환
        df_inputs=pd.DataFrame.from_dict(json_inputs, orient='columns')
        return (df_inputs,)

```

```python
from my_custom_input import MyCustomDataframeInput
@env(infer_pip_packages=True)
@artifacts([
   ...
])

class ModelApiService(BentoService):
    # Custom Input Class 사용 방법    
    @api(input=MyCustomDataframeInput(),
         route="v1/json_input/predict",
         batch=True
        )
    def predict_json_input(self, df: pd.DataFrame):
       ...
        return result
```

- 위 코드는 문자열로 받은 request body 데이터를 json 리스트로 변환 후, 데이터 프레임으로 재변환 하는 구조 + 사전 정의에 따라 에러 코드로 예외 처리하는 기능
- 수신한 데이터를 pandas 또는 imageio 라이브러리를 통해 객체화 시킴
- Micro Batching 기능도 제공함
    
    [Adaptive Micro Batching - BentoML documentation](https://docs.bentoml.org/en/latest/guides/micro_batching.html)
    

### 인풋 명세 확인 기능

```python
from model_api_json_input_validator import ModelApiJsonInputValidator
from model_api_json_validator import ModelApiInputValidator
from bentoml.adapters import DataframeInput
 
@env(infer_pip_packages=True)
@artifacts([
   ...
])
class ModelApiService(BentoService):
    # 커스텀 인풋 명세 예시
    @api(input=ModelApiInputValidator(
             http_input_example=[{"feature1":0.0013,"feature2":0.0234 ... }],
             request_schema= { 
                  "application/json": {
                  "schema": {
                     "type": "object",
                        "required":  ["feature1", "feature2", ... ],
                        "properties": {
                          "feature1": {"type": "float64"}, "feature2": {"type": "float64"}, ...
                         },
                      },
                    }
                 }),
         output=JsonOutput(),
         route="v1/custom_input/predict",
         mb_max_latency=200,
         mb_max_batch_size=500,
         batch=False
        )

    def predict_custom_input(self, df: pd.DataFrame):
        ...
        return result
 

    # 데이터 프레임 인풋 명세 예시
    @api(input=DataframeInput(
            orient = "records",
            colums = ['feature1','feature2', ... ],
            dtype =  {"feature1":"float64", feature2":"float64", ... }),
         output=JsonOutput(),
         route="v1/dataframe_input/predict",
         mb_max_latency=200,
         mb_max_batch_size=500,
         batch=False
        )
    def predict_dataframe_input(self, df: pd.DataFrame):
        ...
        return result
```

- Input 명세에 따른 Example Value 와 Schema 정의됨

![Untitled](Part%201%20%E1%84%91%E1%85%B3%E1%84%85%2079c6a/Untitled%201.png)

### 배치 기능

- 온라인 서빙 뿐만 아니라 배치(오프라인) 서빙도 당연히 가능
- `mb_max_batch_size`, `mb_max_latency` 두 파라미터로 조절

```python
@env(infer_pip_packages=True)
@artifacts([
    KerasModelArtifact('keras_model'),
    XgboostModelArtifact('xgb_model'),
    PytorchModelArtifact('pytorch_model')
])
class ModelApiService(BentoService):
        
    @api(input=ModelApiInputValidator(),
         output=JsonOutput(),
         route="v1/service_name/batch",         
         mb_max_latency=200,     # micro batch 관련 설정
         mb_max_batch_size=500,  # micro batch 관련 설정
         batch=True              # batch 관련 설정
        )
    def predict_v1_batch(self, df: pd.DataFrame):
        ...
        # keras_model 예측
        keras_output = self.artifacts.keras_model.predict(df)
        
        # xgb_model 예측 
        xgb_output = self.artifacts.xgb_model.predict(xgb.DMatrix(df))
        # pytorch_model 예측 
        pytorch_output = self.artifacts.pytorch_model(torch.from_numpy(df))
        # 예측 결과 후처리( 최소 O(n)의 작업 )
        
        # 결과 Return
        return result
```

- **성능 관점에서 대용량 배치는 Airflow와 같은 스케줄러가 더 효율적**

### 패키징 및 Dev 환경 테스트 기능

- 위에서 이용된 python 스크립트, MLFlow를 통해 학습된 모델, 서빙 API 명세, 메타데이터 등 여러 정보를 하나의 파일로 압축하는 과정

```python
# BentoML service packaging
from model_api_service import ModelApiService
model_api_service = ModelApiService()
model_api_service.pack("keras_model", {"model": keras_reconstructed_model, "custom_objects": custom_objects})
model_api_service.pack("xgb_model", xgb_reconstructed_model)
model_api_service.pack("pytorch_model", pytorch_reconstructed_model)

# BentoML Package Upload to yatai server
saved_path = ensemble_churn_predict_service.save()

# dev server start
ensemble_churn_predict_service.start_dev_server(port=5000)

# dev server stop
ensemble_churn_predict_service.stop_dev_server()

# Send test request to dev server
import requests
response = requests.post("http://127.0.0.1:5000/v1/service_name/predict", data='[{"feature1":0.0013803629264609932,"feature2":0.023466169749836886, ... }]')
print(response)
print(response.text)
```

- pack API 로 로컬에 존재하는 모델 패키징
- ML 파이프라인에서 사용하기 위해 패키징된 파일을 BentoML의 모델 관리 컴포넌트인 yatai 서버에 업로드

### 로깅

- BentoML에서는 별도 로그 관리 안함
→ FileBeat를 이용해 별도로 로그 수집하는 구조 선택했음
→ 경량이자, 쉽게 로그 수집 가능
- BentoML에서는 /home/bentoml/logs/* 하위에 남김

```python
- **active.log** – BentoML CLI 로그 혹은 Python 자체에서 남기는 로그
- **prediction.log** – 인풋 요청에 대해 추론한 결과 로그(몇 시에, 어떤 request_id로, 어떤 인풋과 어떤 아웃풋이 나갔는지 로깅)
- **feedback.log** – 추론 결과에 대한 피드백 로그
```

- 비즈니스와 연관된 추가 로깅을 원할 경우 API BentoService 클래스에 아래 코드 작성
→ getLogger("bentoml") 의 경우 active.log 파일에 남음

```python
import logging
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.info("****** bento ml info log *******")
```

- Filebeat 기반 로그 수집 환경을 구성할 경우 Dockerfile 에 아래 코드 추가
+ [docker-entrypoint.sh](http://docker-entrypoint.sh) 에 Filebeat 실행하는 명령어 추가 (컨테이너 띄울 시 무조건 구동)

```python
# 기존 DocekerFile 내용
# logging process를 위한 추가 된 부분
RUN wget https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-{version}-linux-x86_64.tar.gz -O /home/bentoml/filebeat.tar.gz
RUN tar -zxvf /home/bentoml/filebeat.tar.gz
RUN mv filebeat-{version}-linux-x86_64 filebeat
COPY --chown=bentoml:bentoml filebeat_to_secure_kafka.yml ../

# 기존 DocekerFile 내용
USER bentoml
RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT [ "./docker-entrypoint.sh" ]
CMD ["bentoml", "serve-gunicorn", "./"]
```

### 모니터링 기능

- Bentoml 자체적으로 Prometheus metrics API  제공
→ Prometheus 환경만 구성되어있음녀 쉽게 대시보드 구성 가능
→ endpoint와 http_response_code 로 구분하면 총 유입 요청 및 마이크로 배치 요청 양, 에러 비율 확인 가능

![Untitled](Part%201%20%E1%84%91%E1%85%B3%E1%84%85%2079c6a/Untitled%202.png)

### ※ 추가 파이썬 클래스 설명

1. Enum
- enumeration 처럼 인덱스를 이용해 키 밸류로 묶어줌

```python
from enum import Enum

class Skill(Enum):
    HTML = 1
    CSS = 2
    JS = 3

-----

>>> Skill.HTML
<Skill.HTML: 'HTML'>
>>> Skill.HTML.name
'HTML'
>>> Skill.HTML.value
1
```

1. Exception
-

```python
class BigNumberError(Exception): # 사용자 정의 에러
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

try:
    print("한 자리 숫자 나누기 전용 계산기입니다.")
    num1 = int(input("첫 번째 숫자를 입력하세요: "))
    num2 = int(input("두 번째 숫자를 입력하세요: "))
    if num1 >= 10 or num2 >= 10: # 입력받은 수가 한 자리인지 확인
        raise BigNumberError("입력값 : {0}, {1}".format(num1, num2)) # 자세한 에러 메시지
    print("{0} / {1} = {2}".format(num1, num2, int(num1 / num2)))

except ValueError:
    print("잘못된 값을 입력하였습니다. 한 자리 숫자만 입력하세요.")

except BigNumberError as err:
    print("에러가 발생하였습니다. 한 자리 숫자만 입력하세요.")
    print(err) # 에러 메시지 출력
```