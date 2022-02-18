# 1-3. Model Serving (Seldon Core)

![Untitled](1-3%20Model%20%20d6c06/Untitled.png)

![Untitled](1-3%20Model%20%20d6c06/Untitled%201.png)

![Untitled](1-3%20Model%20%20d6c06/Untitled%202.png)

![Untitled](1-3%20Model%20%20d6c06/Untitled%203.png)

- 다양한 서빙 Tools

![Untitled](1-3%20Model%20%20d6c06/Untitled%204.png)

- minikube 에서

---

### Custom Resource

- custom resource (CR) 은 K8S API 확장판
    - 기본 관리 리소스 : `Pod` , `Deployment` , `Service` , `PersistentVolume` 등
    - `User`가 직접 정의한 리소스를 `K8S 의 API` 를 이용해 관리하고 싶은 경우 `Custom Resource` 와 해당 `CR` 의 `LifeCycle` 과 동작을 관리할 `Controller (혹은 API Server)` 를 구현 후 `K8s 클러스터`에 배포해야 함
        - `CR` 을 클러스터에 등록하는 방법은 `Custom Resource Definition (CRD)` 방식과 `API Aggregation (AA)` 방식 두 가지가 존재, 여기선 `CRD` 만 이용
        - `CRD` ß방식은 CR 을 관리할 `Custom Controller` 를 (서드 파티 형식으로) 구현하고 배포하여 사용하게 되며, `Controller` 는 대부분 `Operator Pattern` 으로 개발
- 즉, `K8s` 에서 `Default` 로 관리하지는 않지만, 배포된 `**Custom Controller**` 에 의해 `**k8s**` 에서 관리되고 있는 `리소스`들이라 할 수 있음

---

### Operator Pattern

- Controller
    - `Desired State` 와 `Current State` 를 비교하여, `Current State` 를 `Desired State` 에 **일치시키도록 지속적으로 동작하는 무한 루프**

- Operator
    - `Controller Pattern` 을 사용하여 **사용자의 애플리케이션을 자동화**하는 것
        - 주로 CR 의 `Current/Desired State` 를 지속적으로 관찰하고 `일치`시키도록 동작하는 역할을 위해 사용

- Operator 개발 방법
    - `Operator` 개발에 필요한 부수적인 작업이 자동화되어있는 `Framework`를 활용하여 개발
        - `Kubebuilder`, `KUDO`, `Operator SDK`
    - 앞으로 다룰 `Seldon-core`, `prometheus`, `grafana`, `kubeflow`, `katib` 를 포함해 k8s 생태계에서 동작하는 많은 모듈들이 이러한 `Operator` 로 개발

---

### Helm

- K8s 모듈의 `Package Managing Tool`
    - `Ubuntu OS` 의 패키지 관리 도구 `apt`, `Mac OS` 의 패키지 관리 도구 `brew`, `Python` 의 `pip` 와 비슷한 역할
- 하나의 k8s 모듈은 다수의 리소스들을 포함하는 경우 많음
    - 즉, `a.yaml`, `b.yaml`, `c.yaml`,... 등 많은 수의 k8s 리소스 파일들을 관리해야 하기에 버전 관리, 환경별 리소스 파일 관리 등이 어려움.
- `**Helm**` 은 이러한 작업을 **템플릿화**시켜 **많은 수의 리소스들을 하나의 리소스 처럼 관리**할 수 있게 도움
    - `Helm manifest` 는 크게 `templetes` 와 `values.yaml` 로 이루어짐, `templetes` 폴더에는 해당 모듈에서 관리하는 모든 쿠버네티스 리소스들의 템플릿 파일이 보관
    - manifest file : [컴퓨팅](https://ko.wikipedia.org/wiki/%EC%BB%B4%ED%93%A8%ED%8C%85)에서 집합의 일부 또는 논리정연한 단위인 파일들의 그룹을 위한 [메타데이터](https://ko.wikipedia.org/wiki/%EB%A9%94%ED%83%80%EB%8D%B0%EC%9D%B4%ED%84%B0)를 포함하는 파일
    - `values.yaml` 이라는 인터페이스로부터 사용자에게 값을 입력받아 `templetes` 의 정보와 merge 하여 배포됨

---

### Seldon Core

![Untitled](1-3%20Model%20%20d6c06/Untitled%205.png)

```bash
## install 

wget <URI>

# 압축 풀기
tar -zxvf helm-<virsion>

# 바이너리 Path 로 이동
mv darwin-amd64/helm /usr/local/bin/helm

# helm 정상 동작 확인
helm help
```

![Untitled](1-3%20Model%20%20d6c06/Untitled%206.png)

```bash
## anbassador install

helm repo add datawire https://www.getambassador.io

# helm repo update
helm repo update

# helm install ambassador with som configuration
helm install ambassador datawire/ambassador \
	--namespace seldon-system \
	--create-namespace \
	--set image.repository=quay.io/datawire/ambassador \
	--set enableAES=false \
	--set crds.keep=false

# 정상 설치 확인
kubectl get pod -n seldon-system -w
kubectl get pod -n seldon-system
```

![Untitled](1-3%20Model%20%20d6c06/Untitled%207.png)

![Untitled](1-3%20Model%20%20d6c06/Untitled%208.png)

### Seldon-core install

```bash
helm install seldon-core seldon-core-operator \
> --repo https://storage.googleapis.com/seldon-charts \
> --namespace seldon-system \
> --create-namespace \
> --set usageMetrics.enabled=true \
> --set ambassador.enabled=true
```

![Untitled](1-3%20Model%20%20d6c06/Untitled%209.png)

```bash
kubectl get pod -n seldon-system ## 해당 명령어로 확인
```

![Untitled](1-3%20Model%20%20d6c06/Untitled%2010.png)

---

### Quick Start

- `SeldonDeployment` : `Seldon-Core` 에서 정의한 `Custom Resource` 중 하나
    - `K8s` 에서 `erving 하는 Server 를 `SeldonDeployment` 라 부름
    - Flask 를 사용하는 경우, “API, IP, PORT 를 정의하거나, API 문서를 작성하는 작업 ~ K8s 배포”까지 **필요했던 docker build, push, pod yaml 작성 후 배포 같은 작업없이**, `trained model` 파일이 저장된 경로만 전달하면 자동화된 것이라 볼 수 있음

- seldon 이라는 namespace 생성

```bash
kubectl create namespace seldon
```

- SeldonDeployment YAML 파일 생성

`sample.yaml`

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-model
  namespace: seldon
spec:
  name: iris
  predictors:
  - graph:
      implementation: SKLEARN_SERVER # seldon core 에서 sklearn 용으로 pre-package 된 model server
      modelUri: gs://seldon-models/v1.11.0-dev/sklearn/iris # seldon core 에서 제공하는 open source model - iris data 를 classification 하는 모델이 저장된 위치 : google storage 에 이미 trained model 이 저장되어 있습니다.
      name: classifier
    name: default
    replicas: 1 # 로드밸런싱을 위한 replica 개수 (replica 끼리는 자동으로 동일한 uri 공유)
```

- `modelUri` 에 모델이 저장되어있어야함

- `SeldonDeployment` 생성

```bash
kubectl apply -f sample.yaml
```

- `seldondeployment` 확인

```bash
kubectl get seldondeployment -n seldon
```

- `ambassador` `external IP` 확인

```bash
kubectl get service -n seldon-system
```

- `http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/` 를 통해 접속 가능

![Untitled](1-3%20Model%20%20d6c06/Untitled%2011.png)

### send API Request

- with `curl`

```bash
curl -X POST http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/predictions \
  -H 'Content-Type: application/json' \
  -d '{"data": { "ndarray": [[1,2,3,4]] } }'
```

![Untitled](1-3%20Model%20%20d6c06/Untitled%2012.png)

- 대충 2일 확률이 가장 높다는 뜻

- **with Python Client**

```bash
pip3 install numpy, seldon-core
```

- 본인은 python3 이용하고 싶어서 pip3 로 설치함

`vi test.py`

```python
import numpy as np

from seldon_core.seldon_client import SeldonClient

sc = SeldonClient(
		gateway="ambassador",
		transport="rest",
		gateway_endpoint="192.168.1.111:80",  ## 본인 endpoint 로 이용
		namespace="seldon",
)

client_prediction = sc.predict(
		data=np.array([[1, 2, 3, 4]]),
		deployment_name="iris-model",
		names=["text"],
		payload_type="ndarray",
)

print(client_prediction)
```

![Untitled](1-3%20Model%20%20d6c06/Untitled%2013.png)

- 결과는 다음과 같이 나옴