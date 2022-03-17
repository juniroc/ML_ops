
## Yatai-BentoML

- On-premise K8s 에 BentoML을 띄우기 위해 Yatai-bentoml 을 이용

### why?
Yatai 가 K8s native 하기 때문

- 이때 사내 On-premise K8s 를 이용하기 때문에 문서에 나온 \
`http://yatai.127.0.0.1.sslip.io` 를 이용하면 안됨

- 대신 Port-forwarding 을 통해 접근

![image](/uploads/bc101665bdff66f2a0129889c08f8cc2/image.png)
- `yatai-system` namespace 에서 `yatai` 라는 service 를 열어주면 된다

- initial id : `admin` / initial password : `admin`

접속을 하면 다음과 같이 나옴

![image](/uploads/474d4621b4f248eba15e9bf5dac5e623/image.png)

![image](/uploads/e0746b8d2a8c57d010819d2a655e45bb/image.png)

![image](/uploads/b5b90bb6dcae1c7d37045624f3d8d84e/image.png)

![image](/uploads/ab1c59f2d9cee5d19f865d8fb24a519b/image.png)



`model_packing_save.py`
```
import torch
import lightgbm as lgb
import pandas as pd
import bentoml
import xgboost

# get_model
mo_ = torch.load('./models/xgb_model.pth.tar')['model']

# dr_lauren_classifier와 'model'로 패키징됨
bentoml.xgboost.save('xgb_model', mo_)

```
- 이때 반드시 xgboost 나 Lgbm 등 로드하는 모델의 패키지 버전을 학습할 때와 똑같이 맞춰야 한다. \
-> 이걸로 삽질을 상당히 오래함..

- 모델 저장 python 파일 작성
- 기존 보다 단순해짐 해당 모델에 맞는 라이브러리 불러와서 bentoml.{library}.save('{save_name}', model) 로 작성 가
- `python3 model_packing_save.py` 으로 실행 

![image](/uploads/03dfd88f7f6d05a007bd1c7e5017ac0f/image.png)

- 위와 같이 실행 완료 메시지가 나오며
- `~/bentoml/models` 를 확인해보면 `save_name` 으로 폴더가 생성된 것을 확인 가능
![image](/uploads/4696ba1169d7c42bc95171d40abd6934/image.png)

`bentoml directory tree`

![image](/uploads/221b2135a3b405a370b5eb1a4457e2a8/image.png)

`model.yaml`
```
name: xgb_model
version: 47bkkuve6kwnbkfb
bentoml_version: 1.0.0a6
creation_time: 2022-03-16 06:32:51.898751+00:00
api_version: v1
module: bentoml._internal.frameworks.xgboost
context:
  framework_name: xgboost
  pip_dependencies:
  - xgboost==1.4.2
  bentoml_version: 1.0.0a6
  python_version: 3.8.0
labels: {}
options: {}
metadata: {}
```

- `bentoml models list` 명령어로도 확인 가능
![image](/uploads/a03765bc61ea84504717668ae481db54/image.png)


`service.py`
```
import numpy as np
import pandas as pd
import bentoml
import xgboost
from bentoml.io import NumpyNdarray, PandasDataFrame

# Load runner for the latest lightgbm model 
xgb_runner = bentoml.xgboost.load_runner("xgb_model:latest")

# Creating 'lgbm_classifier' service 
xgb_model = bentoml.Service("xgb_classifier", runners=[xgb_runner])

# Create API function and setting input format
@xgb_model.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(input_arr):
    res = xgb_runner.run_batch(input_arr)  
    return res
```

![image](/uploads/46b8d80ae0d5917884aef6528d4bc8c0/image.png)
- 위처럼 `service.py` 파일을 실행할 필요는 없음

- `bentoml serve ./service.py:xgb_model --reload` 명령어를 통해 모델 서

![image](/uploads/468d121887d6e7ceeb755f7904b38b14/image.png)

- 위와같이 나오면 Serving 이 완료된 것

- `request.py` 파일을 생성하여 REST로 요청하는 모듈을 생성해준다.

`request.py`
```
import requests
import pandas as pd

df= {'time' : [2.78333, 3.183333, 4.4167775, 5.483313, 15.522422, 18.652444], 
     'weekday' : [2, 3, 4, 5, 5, 1], 
     'weekend' : [0, 1, 1, 0, 1, 1], 
     'instlo_1': [3, 12, 3, 11, 11, 11],
     'instlo_2': [128, 55, 37, 34, 45, 23],
     'inst_code' : [142, 1215, 133, 44, 13, 23], 
     'sysname_lo': [1515, 552, 2100, 113, 133, 799], 
     'sysname_eq' : [0, 0, 0, 0, 0, 1]}
df_ = pd.DataFrame(df)

arr = df_.to_numpy()[0]

print(df_)
response = requests.post(
    "http://127.0.0.1:3000/predict",
    headers={"content-type": "application/json"},
    data=df_.to_json())

print(response.text)
```

- 이때 우리는 DataFrame 형식으로 제출하기로 했지만 data에 df_.to_json() 으로 변환해서 넣어주어야 함.

- `python3 request.py` 명령어 실행

![image](/uploads/74a5c389c5214e4a2afc4767218b16bb/image.png)

- 위와 같이 결과가 return 되는 것을 확인할 수 있음

`serving server log`
![image](/uploads/9a5fd3c28958ea771afe88f2468575f0/image.png)
- 서빙하는 곳에서 로그 확인도 가능하다

- 끝.. 난줄 알았지만 이제 시작.. Containerize 하여 K8s 위에 올려야 한다.


- 모델을 서비스에서 바로 서빙하는 방법 말고 벤토로 빌드시켜놓을 수도 있음.

`bentofile.yaml`
```
service: "service.py:xgb_model"
description: "file: ./README.md"   # 해당내용은 설명 내용을 적어둠
labels:
    owner: mj-lee
    stage: demo
include:
  - "*.py"   # bentoml 에 어떤 파일을 넣을지 매칭시킬 패턴 (이후 벤토파일로 해당 패턴과 일치하는 파일들이 옮겨짐 뒤에 캡쳐 확인)
python:
  packages:    # 사용할 패키지
    - scikit-learn
    - pandas
    - xgboost
    - numpy
    - torch
```

- `bentofile`을  작성했다면
- `bentoml build` 명령어 실행

![image](/uploads/73d2332466a7898005b855fdfd40f6d2/image.png)

- 위와같이 성공했다면, `bentoml list` 명령어 를 통해 `bento list`를 확인

![image](/uploads/45d16695ded41f2099be0edcf8d478eb/image.png)

- `~/bentoml/bentoms` 디렉토리를 확인해보면 다음과 같은 구조로 생성되어있음
```
bentoml
├── bentos
    └── xgb_classifier
        ├── latest
        └── y7xubtvfuwwnbkfb
            ├── apis
            │   └── openapi.yaml
            ├── bento.yaml
            ├── env
            │   ├── conda
            │   ├── docker
            │   │   ├── Dockerfile
            │   │   ├── entrypoint.sh
            │   │   └── init.sh
            │   └── python
            │       ├── requirements.lock.txt
            │       ├── requirements.txt
            │       └── version.txt
            ├── models
            │   └── xgb_model
            │       ├── 47bkkuve6kwnbkfb
            │       └── latest
            ├── README.md
            └── src
                ├── model_packing_save_2.py
                ├── model_packing_save.py
                ├── pack_.py
                ├── request.py
                ├── service_2.py
                ├── service.py
                └── test.py
```
- 우선 최하위 directory 인 `src`에 패턴과 일치하는 파일들이 옮겨져 있음
- `bentofile`에서 작성해준 `README.md` 파일 생성
- `models` directory 는 이전에 `service.py`를 그냥 실행했을 떄와 같은 `directory` 생성
- `docker` directory를 보면 이후에 컨테이너화 할 때 쓰일 `Dockerfile` 존재