# Part 2. Kubernetes 기반 적용 예시 및 성능 테스트

### 아키텍쳐

- Native K8s 이용중 → Nginx-Ingress Controller 사용

![Untitled](Part%202%20Kub%20d9fb1/Untitled.png)

### 무중단 배포

- BentoML 과 직접적인 관련내용은 아님
→ K8s 기반 서빙 적용시 꼭 필요
- 잦은 배포를 통해 모델 업데이트는 BentoML 적용의 큰 목적 중 하나
→ 배포를 자주하기 위해서 무중단 배포가 필수

### 배포방법 2가지

1. 마이너 업데이트
- 인풋 피처는 변경되지 않고 모델 또는 API 내부 로직만 변경되어 서빙 API만 배포하는 경우
2. 메이저 업데이트
- 인풋 피처부터 모델까지 모두 변경되면서 서빙 API를 호출하는 클라이언트까지 배포하는 경우

`deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "app.name" . }}
  namespace: {{ .Release.Namespace }}
  labels:
    app: {{ include "app.name" . }}
spec:
  replicas: {{ include "deployment.replicas" . }}
  minReadySeconds: 10
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
    type: RollingUpdate
  selector:
    matchLabels:
      app: {{ include "app.name" . }}
      color: {{ include "color" . }}
  template:
    metadata:
      labels:
        app: {{ include "app.name" . }}
        color: {{ include "color" . }}
    spec:
      containers:
      - name: {{ include "app.name" . }}
        image: {{ include "image.repo.url" . }}
        args: ["bentoml", "serve-gunicorn","--workers","4","./"]
        resources:
          requests:
            memory: {{ include "resource.requests.memory" . }}
            cpu: {{ include "resource.requests.cpu" . }}
          limits:
            memory: {{ include "resource.limits.memory" . }}
            cpu: {{ include "resource.limits.cpu" . }}
        ports:
        - containerPort: {{ include "webserver.port" . }}
```

`service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "service.name" . }}
  namespace: {{ .Release.Namespace }}
spec:
  selector:
    app: {{ include "app.name" . }}
    color: {{ include "color" . }}
  type: NodePort
  ports:
  - protocol: TCP
    port: 443
    nodePort: {{ include "node.port" . }}
    targetPort: {{ include "webserver.port" . }}
```

`ingress.yaml`

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: {{ include "ingress.name" . }}
  namespace: {{ .Release.Namespace }}
spec:
  tls:
  - secretName: {{ include "secret.name" . }}
  rules:
  - host: {{ include "ingress.host" . }}
    http:
      paths:
      - path: /v1
        backend:
          serviceName: {{ include "service.name" . }}
          servicePort: 443
```

`secret.yaml`

```yaml
apiVersion: v1
data:
  tls.crt: {{ include "secret.crt" . }}
  tls.key: {{ include "secret.key" . }}
kind: Secret
metadata:
  name: {{ include "secret.name" . }}
  namespace: {{ .Release.Namespace }}
type: kubernetes.io/tls
```

---

### 롤링 업데이트

- API 파드(pod)가 여러개 일 때, **특정 파드 개수 혹은 비율만큼 순차적**으로 **신규 API 파드로 전환**한다는 의미

위 `deployment.yaml 파일` 의 `.spec.strategy.type : RollingUpdate` 구문으로 적용 가능
`maxSurge, maxUnavailable` 값을 이용해 제어 가능

```yaml
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
    type: RollingUpdate
```

이 예시에서는 **1개의 파드씩 신규 버전의 API로 전환**하면서, 동시에 사용 불가능한 상태의 파드는 허용하지 않겠다는 의미 

- 마이너 업데이트는 무중단으로 가능하나, 메이저 업데이트의 경우 파라미터가 바뀔 수 있어 어려울 수 있음

```yaml
# 신규 버전 Serving API 배포 방법(Docker Image 교체)
kubectl set image deployment {deployment name} {container name}={model_api_docker_image}:{new_model_api_docker_image_tag}

# deployment rollout 상태
kubectl rollout status deployment {deployment name} 

# deployment rollout 히스토리
kubectl rollout history deployment {deployment name}

# 특정 과거 버전 deployment 롤백
$ kubectl rollout undo deployment {deployment name} --to-revision={target revision}
```

---

### 블루/그린 업데이트

- **신규 서빙 API deployment를 추가로 배포한 상태**에서 기존 서빙 API deployment를 바라보고 있던 서비스를 신규 버전으로 전환하는 방법
- deployment가 두 개 배포되어야해 자원 관리에서 문제 생길 수 있으나, 배포 시 서비스만 교체하면 된다는 점에서는 효율적
→ 이슈 발생 시 서비스만 변경하면 되는 구조라 롤백도 편리
- 마이너 업데이트 시 무중단 가능하나, 메이저 업데이트 시 클라이언트 배포가 필수이며, 클라이언트와 서빙 API 배포 시점을 조율해야 함

```yaml
# 기존 버전 Serving API (color - blue)
# 신규 버전 Serving API Deployment 배포 (color - green)
kubectl apply -f patch-deployment.yaml

# 신슈 버전 Serving API 상태 확인
kubectl get deployment
kubectl get pod

# service 전환 방법
patch-service.yaml
spec:
  selector:
    color: green

# 신규 Deployment(color - green)로 서비스 전환
kubectl patch service {service name} -p "$(cat patch-service.yaml)"
```

---

### 카나리

- 신규 서빙 API를 특정 비율만큼만 배포해 테스트 진행하다 문제 없을 경우 점진적으로 전체 서빙 API를 신규 버전으로 전환하는 방법
- 마이너 업데이트인 경우만 적용 가능
→ 메이저 업데이트 시에는 적절하지 않은 방법
- 카나리 방법은 다양
1. 배포되는 파드의 비율을 조정
2. 신규 deployment에 레플리카셋(Replica Set) 값을 적게 설정해 같은 서비스에 연동 후 문제 없을 경우 기존 Deployment의 Replica Set 수를 줄이는 방법 (반대로 신규는 늘림)
→ 문제가 있을 경우 신규 버전 Deployment 를 제거

```yaml
# 신규 버전 Deployment 배포 및 scale 조정
kubectl apply -f new-deployment.yaml
kubectl scale deployment {new deployment} --replicas={increasse replica set number}

# 기존 버전 Deployment 배포 및 scale 조정
kubectl scale deployment {old deployment} --replicas={decrease replica set number}
```

### 인그레스 라우팅

- 전통적인 배포 방법 블루/그린 이나 카나리와 비슷
- 신규와 기존 서빙 API의 Deployment와 Service가 모두 배포된 상태에서 Ingress라우팅 정보만 수정해 각 서비스로 라우트하는 구조
→ 마이너, 메이저 업데이트 모두 효율적으로 배포 가능
- 신규와 기존 서빙 API에 동시 접근 가능한 구조 → 이슈 발생 시 클라이언트에서 엔드포인트 URL만 변경하면 됨
→ 단 Deployment와 Service 모두 각 두벌씩 배포하는 구조라 자원 관리에서 문제 발생

```yaml
# 신규 버전 Deployment 배포
kubectl apply -f new_version_deployment.yaml

# 신규 버전 Service 배포
kubectl apply -f new_version_service.yml

# 수정된 Ingress 파일
patch-ingress.yaml
spec:
  rules:
  - host: {{ include "ingress.host" . }}
    http:
      paths:
      - path: /v1 # API Version에 따라 Routing
        pathType: Prefix
        backend:
          serviceName: {{ include "service.name" . }}
          servicePort: 443
      - path: /v2 # API Version에 따라 Routing
        pathType: Prefix
        backend:
          serviceName: {{ include "service.name.new_version" . }}
          servicePort: 443

# 신규 버전 Ingress 
kubectl patch ingress {service_name} -p "$(cat patch-ingress.yaml)"
```

- 각 서비스에 맞는 방법을 택하면 됨.

※ Argo 같은 배포 도구를 활용하면 보다 효율적인 CD(Continuous Deployment) 구성 가능