# Theory

### 1. Data Scientists' pain points

- Reproducibility is a major concern - best model
→ Traceability become paramount

- **ML Ops - 운영 팀과 기계 학습 연구원 간의 기계 학습 모델 관리, 물류 및 배포를 단순화하는 데 도움
→ 모델 개발, 트레이닝, 배포를 일련의 흐름처럼 지속하게 이루어지게 하는 환경/방법론 
ex) 데이터가 들어오면 전처리 작업 → 학습 → 배포**

![Theory%205bb35/Untitled.png](Theory%205bb35/Untitled.png)

[머신러닝 오퍼레이션 자동화, MLOps](https://zzsza.github.io/mlops/2018/12/28/mlops/)

[ML Ops ~ 거스를 수 없는 대세? 데이터 과학자와 AI 개발자가 쿠버네티스와 컨테이너에 관심을 갖는 이유](https://www.udna.kr/post/ml-ops-~-%EA%B1%B0%EC%8A%A4%EB%A5%BC-%EC%88%98-%EC%97%86%EB%8A%94-%EB%8C%80%EC%84%B8-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B3%BC%ED%95%99%EC%9E%90%EC%99%80-ai-%EA%B0%9C%EB%B0%9C%EC%9E%90%EA%B0%80-%EC%BF%A0%EB%B2%84%EB%84%A4%ED%8B%B0%EC%8A%A4%EC%99%80-%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88%EC%97%90-%EA%B4%80%EC%8B%AC%EC%9D%84-%EA%B0%96%EB%8A%94-%EC%9D%B4%EC%9C%A0)

### Machine Learning Lifecycle

![Theory%205bb35/Untitled%201.png](Theory%205bb35/Untitled%201.png)

 

**Dev Ops  vs  ML Ops**

![Theory%205bb35/Untitled%202.png](Theory%205bb35/Untitled%202.png)

- ML system 도입 전에는 빠르게 구축하긴 했지만, 시간이 지나면서 유지보수하려면 어려움
→ 도입 이후에는 보다 빠르게 유지보수 가능

![Theory%205bb35/Untitled%203.png](Theory%205bb35/Untitled%203.png)

 

### Phases of a ML project

1. **Discovery phase** 
- establish the problem or task
2. **Development phase**
3. **Deployment phase**

![Theory%205bb35/Untitled%204.png](Theory%205bb35/Untitled%204.png)

Q. 

1) Which platform should host my model?

2) Which service should i pick for model serving? 

3) How many Nodes should the cluster have?

### ML Pipeline 프로세스

![Theory%205bb35/Untitled%205.png](Theory%205bb35/Untitled%205.png)

**Discovery → Development → Deployment**

![Theory%205bb35/Untitled%206.png](Theory%205bb35/Untitled%206.png)

### Intro Containers and Kubernetes

![Theory%205bb35/Untitled%207.png](Theory%205bb35/Untitled%207.png)

- application의 모든 dependency와 Operating system 을 VM 에서 다른곳으로 이동하기 어렵다.
→ 매번 VM을 시작할 때마다 Operating system은 부팅하는데 시간이 걸린다.

   

![Theory%205bb35/Untitled%208.png](Theory%205bb35/Untitled%208.png)

- 여러 Application가 single VM에서 dependency를 공유하는 경우 
→ 한 곳에서 많은 dependency를 요청할 경우 : 다른 app이 stop 될 수 있다.

![Theory%205bb35/Untitled%209.png](Theory%205bb35/Untitled%209.png)

- single  kernel for each Applications 
→ 이것은 낭비가 심하다

![Theory%205bb35/Untitled%2010.png](Theory%205bb35/Untitled%2010.png)

 컨테이너는 User 공간을 독립적으로 이용
→ 공간 이용 효율적이다.
→ Application만 켜고 끄면 됨. VM 전체를 껏다 킬 필요 없음. 

**※ 컨테이너가 필요한 이유.**

### Container and Container images

![Theory%205bb35/Untitled%2011.png](Theory%205bb35/Untitled%2011.png)

위에서 **Application 과 Dependency** 를 **Image** 라함.

- Container는 이미지의 인스턴스를 실행하는 것
- Docker 는 컨테이너가 app을 실행, 생성하는 것을 도움
→ 그러나 Orchestration은 불가능
→ k8s가 가능케 함.

**Containers Foundations**

1. **Processes**
- Each Linux process has its own virtual memory address phase, separate from others
- its are rapidly created and destroyed
2. **Linux namespaces
-** to control what an application can see process ID numbers, directory trees, IP address etc..
3. **Cgroups**
- to control what an application can use its maximum consumption of CPU time, memory, IO bandwidth, other resources.
4. **Union file system**
- to efficiently encapsulate application and their dependencies into a set of clean minimal layers.  

**Container Image is structured in layers**

**How it works**

![Theory%205bb35/Untitled%2012.png](Theory%205bb35/Untitled%2012.png)

**A container image is structured in layers**

**Container Image 형식의 Docker** 를 **Docker file** 이라 부름.

**1st layer(Pull, 사진의 맨아래)
- pulled from public repository 
- Ubuntu 특정 버젼에서 이루어 짐**

**2nd layer (Copy)
- copied layer
- containing files from build tools current directory**

**3rd layer (Run)
- build application using the make command and puts the result of the build
- build 결과물을 생성**

**4th layer (Launch)
- specified what command to run within the container when it's launched**

 

---

### K8s  (container management platform)

![Theory%205bb35/Untitled%2013.png](Theory%205bb35/Untitled%2013.png)

- Kubernetes job is to make the deployed system conform to desired state and keep it there in spite of failure
- **Declarative Configuration (선언형 구성)** saves work.
→ because the system is desired state is always documented.
→ it reduce the risk of error
- Imperative Configuration  (명령형 구성)
→ 이곳에서 시스템 상태 변경을 위해 issue command
→ 이 경우 자동으로 시스템 상태를 유지하는 이점을 잘못 이용할 수 있음
→ 능숙한 이용자들은 선언적 구성의 빠른 수정 툴로 이용한다

### K8s **Feature**

1. Supports both stateful and stateless applications
2. Autoscaling
3. Resource limits
4. Extensibility
5. Portability

### GKE (Google Kubernetes Engine)

1. Deploy
2. Manage 
3. Scale k8s env.
- It is a component of GCP

---

### Compute Options Detail

![Theory%205bb35/Untitled%2014.png](Theory%205bb35/Untitled%2014.png)

### **Compute Engine**

- Fully Customizable virtual machines
- Persistent disks and optional local SSDs
- Global load balancing and autoscaling
- Per-second billing

 

![Theory%205bb35/Untitled%2015.png](Theory%205bb35/Untitled%2015.png)

 

### App Engine

- Provides a fully managed, code-first platform
- Streamlines application deployment and scalability
- Provides support for popular Programming language and application runtimes.
- Supports integrated monitoring, logging, and diagnostics.
- Simplifies version control, canary testing, and rollbacks.

→ Deploying보다 Building Application (Coding) 에 집중할 수 있도록 돕는다.

![Theory%205bb35/Untitled%2016.png](Theory%205bb35/Untitled%2016.png)

**RESTful APIs : Application program interface** 
- 웹 브라우저와 웹서버의 관계와 같음
→ 이것은 개발자들이 쉽게 작업하도록 도움

### Google K8s Engine

- Fully managed K8s platform
- Supports cluster scaling, persistent disks, automated upgrades, and auto node repairs.
- Built-in integration with Google Cloud services.
- Portability across multiple environments
- Hybrid computing
- Multi-cloud computing

![Theory%205bb35/Untitled%2017.png](Theory%205bb35/Untitled%2017.png)

### Cloud Run

- Enables stateless containers
- Abstracts away infrastructure management.
- Automatically scales up and down
- Open API and runtime environment

![Theory%205bb35/Untitled%2018.png](Theory%205bb35/Untitled%2018.png)

### Cloud Functions

- Event-driven, serverless compute service.
- Automatic scaling with highly available and fault-tolerant design
- Charges apply only when your code run
- Triggered based on events in Google Cloud services, HTTP endpoints, and Firebase

 

![Theory%205bb35/Untitled%2019.png](Theory%205bb35/Untitled%2019.png)

![Theory%205bb35/Untitled%2020.png](Theory%205bb35/Untitled%2020.png)

ex) 

1. 물리적 서버 하드웨어에서 애플리케이션을 실행하는 경우,
각 VM이 관리되고 유지되는 수명이 긴 가상 머신에서 애플리케이션을 실행하는 경우
→ **Compute Engine**

2. 운영과 관련된 경우
**→ App Engine and Cloud Function**

3. App Engine이 제공하는 것보다 컨테이너화 된 워크로드를 더 많이 제어하려는 경우,
Compute Engine이 제공하는 것보다 더 밀집된 패킹을 제어하려는 경우
**→ GKE**

4. 관리 형 컴퓨팅 플랫폼에서 상태 비 추적 컨테이너를 실행하는 경우
**→ Cloud Run**

![Theory%205bb35/Untitled%2021.png](Theory%205bb35/Untitled%2021.png)

---

### K8s Concepts

- 2 Elements of Kubernetes Objects

![Theory%205bb35/Untitled%2022.png](Theory%205bb35/Untitled%2022.png)

### Pod

![Theory%205bb35/Untitled%2023.png](Theory%205bb35/Untitled%2023.png)

- **Smallest Deployable K8s object**
- **Container는 Pod 안에 있음**
- it shares the network namespace 
ex) IP address, network ports
- 같은 Pod 안의 컨테이너는 서로 communication 가능
→ 127.0.0.1 IP address
- Pods는 스스로 디버깅 불가능하고

![Theory%205bb35/Untitled%2024.png](Theory%205bb35/Untitled%2024.png)

Kubernetes Control Plane은 클러스터의 상태를 지속적으로 모니터링하여 선언 된 것과 Current state를 계속적으로 비교하고 필요에 따라 상태를 수정합니다.

### **K8s Control Plane**

![Theory%205bb35/Untitled%2025.png](Theory%205bb35/Untitled%2025.png)

- VM에 구축된 Cluster 내부 컴퓨터 하나를 Master라 부르고 다른 것들은 Node라 함
→ Node의 역할을 running a pod
→ Master의 역할은 coordinate the entire cluster

**Kube-APIServer
→** pod launching과 클러스터 상태에 관련된 명령을 받아들임
→ 권한 부여

**Kubectl**
→ Kube-API sever와 연결한다 

**etcd**
→ Cluster's Database
→ cluster의 상태(configuration data 및 dynamic information)를 저장

ex) 어떤 노드가 어떤 클러스터의 부분인지, 어떤 pod를 실행해야하는지 등

**Kube-scheduler**

→ Scheduling pods
ex) pod의 요구사항을보고 어떤 노드가 적절한지 선정

**Kube-controller-manager
→** Cluster 상태 monitoring
→ Cluster를 원하는 상태로 만듬  

 

**Kubelet
→** To serve as Kubernetes’s agent on each node

**Kube-proxy**
→ 네트워크 유지

---

### 배포

![Theory%205bb35/Untitled%2026.png](Theory%205bb35/Untitled%2026.png)

1. 업데이트 할때마다 최신 컨테이너 이미지를 사용하도록 할 수 있다.
2. 업데이트 된 포드가 안정적이지 않을 경우 버젼을 롤백할 수 있다.
3. 배포 구성을 수정해 포드를 수동으로 확장 가능 
4. Stateless 애플리케이션은 상태를 클러스터에 저장하지 않는다.
→ 원하는 상태는 배포 YAML 파일에 설명

- 배포는 상태를 선언하는 포드 용 상위 수준 컨트롤러
→ 복제 세트 컨트롤러를 구성하고 인스턴스화, 지정된 특정 버전의 포드를 유지.

---

### 무중단 배포 : 서비스가 중단되지 않으며 배포할 수 있도록 하는 기술

1. Rolling Update
- 새 버전 인스턴스를 하나씩 늘려가고 기존 버전 인스턴스를 하나씩 줄여가는 방식
→ 트래픽이 이전되기 전까지 이전 버전과 새 버전의 인스턴스가 동시에 존재할 수 있다는 단점 
    
    ![Theory%205bb35/Untitled%2027.png](Theory%205bb35/Untitled%2027.png)
    
2. Blue-Green
- 구 버전, 신버전 서버를 동시에 나란히 구성하여 배포 시점에 트래픽이 일제히 전환
→ 시스템 자원이 두배로 필요하여 많은 비용 발생
    
    ![Theory%205bb35/Untitled%2028.png](Theory%205bb35/Untitled%2028.png)
    
3. Canary
- 특정 서버 또는 특정 user에게만 배포했다가 정상적이면 전체를 배포

![Theory%205bb35/Untitled%2029.png](Theory%205bb35/Untitled%2029.png)

---

### JOB

- 예약된 작업
ex) 서버 관리 목적이나 애플리케이션 관리 등의 목적으로 단 한 번의 반복된 형태의 작업을 실행할 수 있는 방법
→ 정해진 Job이 끝나면 Pod는 소멸된다.
→ 서비스가 아닌 작업을 위해 존재하는 컨트롤러

### Cron Job

- 원하는 작업을 예약된 스케쥴링으로 동작시키고 싶을 때 사용하는 컨트롤러
ex) ETL할 때, 실시간으로 들어오는 데이터를 n분에 한 번, n시간에 한 번 작업을 돌리고 싶을 때원하는 작업을 예약된 스케쥴링으로 동작시키고 싶을 때 사용하는 컨트롤러ex) ETL할 때, 실시간으로 들어오는 데이터를 n분에 한 번, n시간에 한 번 작업을 돌리고 싶을 때

---

# 2 Week

![Theory%205bb35/Untitled%2030.png](Theory%205bb35/Untitled%2030.png)

### **AI 솔루션의 3가지 문제**

![Theory%205bb35/Untitled%2031.png](Theory%205bb35/Untitled%2031.png)

1. **배포**
- 인프라의 복잡성이 증가하여 강력한 도구 상자가 필요

    
    ![Theory%205bb35/Untitled%2032.png](Theory%205bb35/Untitled%2032.png)
    
2. **Talent**
- AI 앱을 구축하려면 기준을 낮춰야 할 수도 있다.
→ 재사용되고 누구나 다시 구현할 수 있는 것.

- 계층화를 할 수도 있다.
→ 
1. 처음부터 솔루션 구축까지
2. 1에서 만들어진 것을 사용, 기타 비즈니스 문제에 맞게 사용자 지정
3. 그것을 이용

    
    ![Theory%205bb35/Untitled%2033.png](Theory%205bb35/Untitled%2033.png)
    
3. **협업**

![Theory%205bb35/Untitled%2034.png](Theory%205bb35/Untitled%2034.png)

---

### ML Pipeline

![Theory%205bb35/Untitled%2035.png](Theory%205bb35/Untitled%2035.png)

 

![Theory%205bb35/Untitled%2036.png](Theory%205bb35/Untitled%2036.png)

![Theory%205bb35/Untitled%2037.png](Theory%205bb35/Untitled%2037.png)

![Theory%205bb35/Untitled%2038.png](Theory%205bb35/Untitled%2038.png)

---

### AI Platform 파이프라인

![Theory%205bb35/Untitled%2039.png](Theory%205bb35/Untitled%2039.png)

 

![Theory%205bb35/Untitled%2040.png](Theory%205bb35/Untitled%2040.png)

---

### Pipeline 사용 시기

![Theory%205bb35/Untitled%2041.png](Theory%205bb35/Untitled%2041.png)

![Theory%205bb35/Untitled%2042.png](Theory%205bb35/Untitled%2042.png)

![Theory%205bb35/Untitled%2043.png](Theory%205bb35/Untitled%2043.png)

![Theory%205bb35/Untitled%2044.png](Theory%205bb35/Untitled%2044.png)

### Eco-System

![Theory%205bb35/Untitled%2045.png](Theory%205bb35/Untitled%2045.png)

![Theory%205bb35/Untitled%2046.png](Theory%205bb35/Untitled%2046.png)

---

### 실습

![Theory%205bb35/Untitled%2047.png](Theory%205bb35/Untitled%2047.png)

---

### AI Pipeline system

**process**

![Theory%205bb35/Untitled%2048.png](Theory%205bb35/Untitled%2048.png)

 

![Theory%205bb35/Untitled%2049.png](Theory%205bb35/Untitled%2049.png)

![Theory%205bb35/Untitled%2050.png](Theory%205bb35/Untitled%2050.png)

---

**BigQuery**
- 반복 가능한 데이터 세트 분할 생성
→ 훈련, 검증, 테스트로 분할

![Theory%205bb35/Untitled%2051.png](Theory%205bb35/Untitled%2051.png)