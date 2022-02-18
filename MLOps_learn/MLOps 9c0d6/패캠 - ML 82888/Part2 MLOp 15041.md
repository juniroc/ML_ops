# Part2. MLOps 환경 구축을 위한 Docker, K8S

![Untitled](Part2%20MLOp%2015041/Untitled.png)

![Untitled](Part2%20MLOp%2015041/Untitled%201.png)

![Untitled](Part2%20MLOp%2015041/Untitled%202.png)

![Untitled](Part2%20MLOp%2015041/Untitled%203.png)

![Untitled](Part2%20MLOp%2015041/Untitled%204.png)

- 많은 메모리를 필요로 하는 Container, 많은 GPU를 필요로 하는 Container, 많은 저장공간을 필요로 하는 Container
    
    → 각각의 요구사항에 맞는 서버에 할당 
    

![Untitled](Part2%20MLOp%2015041/Untitled%205.png)

---

### 도커

![Untitled](Part2%20MLOp%2015041/Untitled%206.png)

![Untitled](Part2%20MLOp%2015041/Untitled%207.png)

![Untitled](Part2%20MLOp%2015041/Untitled%208.png)

![Untitled](Part2%20MLOp%2015041/Untitled%209.png)

- ML Engineer 들도 이제는 도커와 K8S를 무조건 다룰 줄 알아야 함.

### 명령어

- 도커 로그 확인

```bash
docker logs <docker name>
```

---

### Dockerfile

 

`Dockerfile`

```bash
FROM ubuntu:18.04   # base 이미지 어떤것을 이용할 것인지 명시

COPY a.txt /directory/b.txt

RUN pip install -r requirments.txt

WORKDIR /home/dr_lauren/pipeline

CMD ["python","main.py","alram"]

ENV LANG_ ko_KR. ## 또는 ENV LANG_=ko_KR.

EXPOSE 8080
```

- From : Base image 명시
- COPY : <scr> 파일을 원하는 디렉토리 위치로 복사하는 것
- RUN : 명시한 커맨드를 도커 컨테이너 내부에서 실행
- WORKDIR : 이후 작성될 명령어를 컨테이너 내부 어느 디렉토리에서 수행할 것인지 명시
- CMD : 커맨드를 도커 컨테이너가 시작될 때, 실행하는 명령어 
`ENTRYPOINT` 와 비슷 (구글링으로 찾아볼 것)
- `하나의 도커 이미지`에서는 `하나의 CMD` 이용 가능
- ENV : `컨테이너 내부`에 `지속적`으로 사용될 `환경 변수` 설정
- EXPOSE : `컨테이너 내부 포트/프로토콜` 설정

### docker build

```bash
docker build -t my-image:v0.0.1 .
```

- `.` : 현재 경로에 있는 `Dockerfile` 을 이용
- `-t` : `my-image` 라는 이름과 `v0.0.1` 이라는 태그를 이용

### Kubernetes

![Untitled](Part2%20MLOp%2015041/Untitled%2010.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2011.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2012.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2013.png)

- K8S - 선언형 인터페이스
- `Kubernetes Native` 하다 라고 표현

![Untitled](Part2%20MLOp%2015041/Untitled%2014.png)

- `master`, `worker` 역할을 하는 `node` 가 분리되어있음
→ 보통 하나 이상의 서버를 묶어서 이용
→  물리적으로 여러대의 컴퓨터가 분리되어 있어도, 한번 가상화가 되어 하드웨어의 소프트웨어화가 됨
→ **여러대의 서버를 한대의 서버처럼 이용** 가능
- `master` 역할을 하는 `Control Plane` , `worker` 역할을 하는 `node` 들로 구성
- `Control Plane` 노드는 여러대의 `worker` 노드를 관리 및 모니터링, `client` 로 부터 요청을 받고, 그 요청에 맞는 `worker node`를 스케줄링해 해당 `node` 로 요청을 전달
- `Client`가 보낸 요청을 받는 곳이 `API Server`
- 사용자 보낸 요청의 `desire statement` 를 `<key>:<value>` 형태로 저장하는 database가 `etcd`
- `Control Plane`  으로 부터 명령을 받고, 다시 `worker node` 의 상태를 전달하는 곳이 `kubelet`

---

### YAML

- 데이터 직렬화에 쓰이는 포멧/양식 중 하나
    - k8s 에서 마스터에게 양식을 보낼 경우, `YAML` 또는 `JSON` 형태로 보내야 함
    - `yml`, `yaml`
- `Key-value` pair 의 집합

![Untitled](Part2%20MLOp%2015041/Untitled%2015.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2016.png)

- List 형태

![Untitled](Part2%20MLOp%2015041/Untitled%2017.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2018.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2019.png)

- 띄어쓰기 관련

### Multi-document Yaml

![Untitled](Part2%20MLOp%2015041/Untitled%2020.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2021.png)

### kubectl

- kubernetes cluster (server)에 요청을 간편하게 보내기 위해 널리 사용되는 client 툴

---

### POD

- K8s 에서 생성 및 관리할 수 있는 **최소 컴퓨팅 단위**
    
    → 즉, **Pod 단위**로 **스케줄링, 로드밸런싱, 스케일링 작업** 수행
    
    → `Pod` 내부 `Container` 들은 자원을 공유
    
    → **stateless**(상태를 저장하지 않음)한 특징을 가지며, 언제든 삭제가 가능한 자원임
    
    ![Untitled](Part2%20MLOp%2015041/Untitled%2022.png)
    

![Untitled](Part2%20MLOp%2015041/Untitled%2023.png)

- K8S resource 의 desired state 기록을 위해 항상 YAML 파일을 저장 및 관리

### Namespace

- K8S에서 리소스를 격리하는 가상의(논리적인) 단위
    
    `kubectl config view --minify } grep namespace` 를 통해 `current namespace`가 어떤 namespace로 설정되었는지 확인할 수 있음
    → 따로 설정하지 않았으면 `default namespace` 가 기본으로 설정됨
    

```bash
kubectl get pod -n kube-system 
# kube-system namespace 의 pod 조회
```

```bash
kubectl get pod <pod-name> -o yaml
# yaml 형식으로 pod 정보 출력

kubectl get pod -w
# pod 결과를 계속보여주며, 변화가 있을 때만 업데이트
```

### Log

```bash
kubectl logs <pod-name>

kubectl logs <pod-name> -f
# 로그를 계속 보여줌
```

- Pod 내부에 container가 여러개 있는 경우

```bash
kubectl logs <pod-name> -c <container-name>

kubectl logs <pod-name> -c <container-name> -f
# 로그를 계속 보여줌 
```

### 내부접속

```bash
kubectl exec -it <pod-name> -- <명령어>

# 주로 <명령어> 는 bash 를 이용해서 접속했음

kubectl exec -it counter -- bash
```

![Untitled](Part2%20MLOp%2015041/Untitled%2024.png)

### 삭제

```bash
kubectl delete pod <pod-name>

kubectl delete -f <YAML-파일-경로>
# 적용했던 YAML로 생성된 pod만 삭제
```

---

### Deployment

- `Pod` 와 `Replicaset` 에 대한 관리를 제공하는 단위
- 관리 : `Self-healing`, `Scaling`, `Rollout(무중단 업데이트)` 같은 기능 포함
- Pod를 감싼 개념, 즉 Deployment로 배포함으로써 **여러 개로 복제된 여러 버전의 Pod**을 안전하게 **관리**하는 것으로 생각하면 편함

![Untitled](Part2%20MLOp%2015041/Untitled%2025.png)

- 사실 위에서 메인은 `spec` 이후로 들어가는 부분이라고 보면 됨

```bash
kubectl apply -f deployment.yaml
# pod 와 비슷한 방법으로 yaml 적용

kubectl get deployment
# deployment 확인

# 또는 pod 이랑 동시에 확인 가능
kubectl get deployment,pod
```

![Untitled](Part2%20MLOp%2015041/Untitled%2026.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2027.png)

- `pod` 3개 모두 `deployment`와 같은 시간에 생성된 것을 확인할 수 있음

- AutoHealing 기능 확인

![Untitled](Part2%20MLOp%2015041/Untitled%2028.png)

- Pod 하나 삭제

![Untitled](Part2%20MLOp%2015041/Untitled%2029.png)

- 그러나 하나가 다시 생성된 것을 확인할 수 있음
→ pod-name 은 바뀜

### Deployment Scaling

- Replica 개수 늘리는 것

```bash
kubectl scale deployment/nginx-deployment --replicas=5

kubectl get deployment

kubectl get pod
```

![Untitled](Part2%20MLOp%2015041/Untitled%2030.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2031.png)

- 당연히 줄이는 것도 가능

### Deployment Delete

```bash
kubectl delete deployment <deployment-name>
```

![Untitled](Part2%20MLOp%2015041/Untitled%2032.png)

- pod와 동일하게 적용된 YAML 파일 경로를 이용해서도 제거 가능

```bash
kubectl delete deployment -f ~.yaml 
```

### Service

- `K8s`에 배포한 `애플리케이션(Pod)`를 `외부에서 접근`하기 `쉽게 추상화한 리소스`
- `Pod` 은 `IP 를 할당`받고 `생성`되지만, **언제든지 죽었다 살아남**
→ 그 과정에서 `IP 는 항상 재할당`
→ 고정된 `IP` 로 원하는 `Pod`에 접근 불가능
→ 즉 클러스터 외부에서 접근할 경우 `Pod` 이 아닌 `Service` 를 통해 접근
- `Service` 는 고정된 IP 가지며, `여러개의 Pod` 와 매칭됨
- `Service`의 주소로 접근하면, 실제로 `Service` 에 매칭된 `Pod` 에 접속

### Service 생성

- 우선 Deployment 재생성 한 후에 적용

![Untitled](Part2%20MLOp%2015041/Untitled%2033.png)

- IP 가 각각 다른 것을 확인할 수 있음

![Untitled](Part2%20MLOp%2015041/Untitled%2034.png)

- `Pod` 하나 삭제 후 (맨 아래꺼) 다시 `IP` 를 확인해보면 변경됨

![Untitled](Part2%20MLOp%2015041/Untitled%2035.png)

- 그냥은 접속이 안됨

```bash
# minikube 안에서는 통신이 가능하긴 함
minikube ssh
# minikube 내부로 접속

curl -X GET <POD-IP> -vvv
ping <POD-ID>
# 통신 기능
```

### Deployment를 매칭한 Service 생성

![Untitled](Part2%20MLOp%2015041/Untitled%2036.png)

`service.yaml`

```bash
apiVersion: v1
kind: Service
metadata:
  name: my-nginx
  labels:
    run: my-nginx
spec:
  type: NodePort # Service 의 Type 명시
  ports:
  - port: 80
    protocol: TCP
  selector:
    app: nginx  # 이 부분을 Deployment와 매칭 시켜줘야 함.
```

![Untitled](Part2%20MLOp%2015041/Untitled%2037.png)

- NodePort - Node Ip 는 그대로 사용, Port 는 따로 사용한다는 뜻

- k8s IP 확인

![Untitled](Part2%20MLOp%2015041/Untitled%2038.png)

```bash
curl -X GET $(minikube ip):<PORT> 
`
 했을 경우 서비스를 통해 외부에서도 정상적으로 
 pod에 접속할 수 있는 것 확인 가능해야됨
`

# 여기서 우리는 210.114.89.130 (미크로틱 IP)로 할당
# 31291 -> 30025 로 열어줬음 (미크로틱에서)
curl -X GET 210.114.89.130:30025
```

### Service Type

- **NodePort** 라는 Type 사용하면, `kuberenetes cluster 내부에 배포된 서비스`에 `클러스터 외부`에서 **접근 가능**
    - 접근하는 IP는 pod 이 떠있는 노드(머신)의 IP를 사용(우리는 210.114.89.130)
    - Port는 할당받은 Port 사용 (여기선 31291)
- **LoadBalancer** Type 도, **외부에서 접근 가능**
    - 단, 이를 사용하려면 `LoadBalancing 역할`을 하는 모듈 추가적으로 필요
- **Cluster_IP** 라는 Type은 고정된 IP, PORT 제공
    - 클러스터 내부에서만 접근할 수 있는 대역의 주소 할당
        
        → 외부에서는 접근 불가
        

### **실무에서는?**

- k8s cluster에 `MetalLB` 같은 `LoadBalancing` 역할을 하는 모듈 설치 후, `LoadBalancer type`으로  **서비스를 expose** 하는 방식 사용
    
    → NodePort 는 **Pod 이 어떤 Node 에 스케줄링될지 모르는 상황**에서, **Pod이 할당된 후 해당 Node의 IP를 알아야 한다는 단점**이 존재
    

---

### PVC

- `Persistent Volume (PV)`, `Persistent Volume Claim(PVC)` 는 stateless 한 Pod 이 영구적으로(persistent) `데이터를 보존` 하고 싶은 경우 사용하는 리소스
    
    → `docker run`의 `-v` 옵션과 비슷
    
- **PV : 관리자가 생성한 실제 저장 공간의 정보를 담음(관리자 입장)
PVC : 사용자가 요청한 저장 공간의 스펙에 대한 정보를 담음(사용자 입장)**
- **즉 `Pod` 는 언제든지 사라질 수 있기에, 보존하고 싶은 데이터 있을 경우, `Pod` 에 `PVC`를 `mount` 해서 사용해야 함.**
→ **PVC**를 사용하면 **여러 Pod 간의 data 공유도 쉽게** 가능

### PVC 생성

- minikube 로 진행할 경우 아래와 같이

![Untitled](Part2%20MLOp%2015041/Untitled%2039.png)

 → `standard` 라는 `storageclass`가 생성되어 있음

- `**PVC` 생성시 해당 `PVC` 스펙에 맞는 `PV`를 그 즉시 `자동으로 생성`해준 뒤, `PVC 와 매칭` 시켜준다고 이해하면 됨.**

`PVC.yaml`

```bash
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: myclaim
spec:
  accessModes:
    - ReadWriteMany  # ReadWriteOnce, ReadWriteMany 옵션 중 선택
  volumeMode: Filesystem
  resources:
    requests:
      storage: 10Mi
  storageClassName: standard
```

![Untitled](Part2%20MLOp%2015041/Untitled%2040.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2041.png)

![Untitled](Part2%20MLOp%2015041/Untitled%2042.png)

### Pod 에서 PVC 사용

- Pod 을 생성
    - volumeMounts, volumes 부분이 추가

`pod-pvc.yaml`

```bash
apiVersion: v1
kind: Pod
metadata:
  name: mypod
spec:
  containers:
    - name: myfrontend
      image: nginx
      volumeMounts:
      - mountPath: "/var/www/html"
        name: mypd
  volumes:
    - name: mypd
      persistentVolumeClaim:
        claimName: myclaim
```

![Untitled](Part2%20MLOp%2015041/Untitled%2043.png)