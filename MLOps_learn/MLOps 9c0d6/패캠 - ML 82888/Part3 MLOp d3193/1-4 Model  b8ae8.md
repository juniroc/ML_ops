# 1-4. Model Monitoring

![Untitled](1-4%20Model%20%20b8ae8/Untitled.png)

![Untitled](1-4%20Model%20%20b8ae8/Untitled%201.png)

- 수 많은 장애 상황이 위와 같이 많은데, 그때그때 모니터링할 수 있는 시스템이 필요함

![Untitled](1-4%20Model%20%20b8ae8/Untitled%202.png)

![Untitled](1-4%20Model%20%20b8ae8/Untitled%203.png)

- 점점 인프라를 개발하는 경우가 더 늘어나고 있음

![Untitled](1-4%20Model%20%20b8ae8/Untitled%204.png)

`The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction, IEEE, Google`

![Untitled](1-4%20Model%20%20b8ae8/Untitled%205.png)

1. Dependency가 result에 주는 영향
2. 학습 및 서빙 Data 불변 관련
3. 학습 및 서빙 feature 가 동일하게 되는지
4. 모델이 너무 old한지
5. 모델이 안정적인지
6. 편향, 모델 속도, 의존성, 메모리 사용량 등 
7. 모델의 회귀 정도

![Untitled](1-4%20Model%20%20b8ae8/Untitled%206.png)

- 분포가 의도한 바와 동일한지 등
- 편향 정도 등
- 리소스 소비 정도
- 모델, 네트워크 안정

→ 이슈해결한 `commit`을 추가하고 새로 배포하고나서 → 제대로 동작하는 것을 확인할 수 있는 `end-to-end`  test 자동화와 FAST API 정도는 구축해 놓아야함.

- ML 서비스 모니터링이 어려운 이유

![Untitled](1-4%20Model%20%20b8ae8/Untitled%207.png)

![Untitled](1-4%20Model%20%20b8ae8/Untitled%208.png)

- loki : loading 을 위한 tool - grafana 와 prometheus 를 연결해 주는 툴
- Thanos : 이러한 모니터링 시스템을 모니터링하는 툴

→ 실무에서 다룰 경우 적용해 볼 것.

---

### Prometheus

- SoundCloud 에서 만든 모니터링 & 알람 프로그램
- 2016년 CNCF 에 Joined, 2018년 Graduated 하여 완전 독립형 오픈소스 프로젝트로 발전
- K8s에 종속적이지 않고, binary 혹은 docker container 형태로도 사용 및 배포 가능
- K8s 생태계의 오픈소스 중에서는 사실상의 표준
    - 구조가 k8s와 궁합이 맞고, 다양한 플러그인이 오픈소스로 제공

### 특징

- 수집하는 Metric 데이터를 다차원의 시계열 데이터 형태로 저장
- 데이터 분석을 위한 자체 언어 PromQL 지원
- 시계열 데이터 저장에 적합한 TimeSeries DB 지원
- **데이터 수집하는 방식이 Pull 방식**
    - 모니터링 대상의 `Agent 가 Server로 Metric을 보내는 Push`방식 이 아닌, `Server 가 직접 정보를 가져가는 Pull` 방식
    - `Push` 방식을 위한 `Push Gateway` 도 지원
- 다양한 시각화 툴과의 연동 지원
- 다양한 방식의 Alarming 지원

![Untitled](1-4%20Model%20%20b8ae8/Untitled%209.png)

- **Prometheus Server**
    - 시계열 데이터를 수집하고 저장
        - `metrics 수집 주기`를 설치 시 정할 수 있으며 `default` 는 15초
- **Service Discovery**
    - `Monitoring` 대상 리스트를 조회
    - `사용자`는 `K8s 에 등록`하고, `Prometheus Server` 는 `K8s Api Server` 에 모니터링 대상을 물어보는 형태
- **Exporter**
    - `Prometheus` 가 `Metrics` 을 수집해갈 수 있도록 정해진 `HTTP Endpoint` 를 제공해 정해진 형태로 `metrics 를 Export`
    - `Prometheus Server` 가 이 `Exporter 의 Endpoint` 로 `HTTP GET Request` 를 보내 `metrics 를 Pull` 하여 저장
    - 하지만, 이런 `Pull 방식`은 **수집 주기와 네트워크 단절 등의 이유**로 `모든 metrics 데이터를 수집하는 것을 보장할 수 없기` 때문에 `Push` 방식을 위한 `Push gateway` 제공
- **Pushgateway**
    - 보통 `Prometheus Server` 의 `pull 주기 (period) 보다 짧은 lifecycle` 을 지닌 작업의 metrics 수집 보장을 위함
- **AlertManager**
    - `PromQL` 을 사용해 `특정 조건문`을 만들고, 해당 **조건문이 만족되면 정해진 방식으로 정해진 메시지**를 보낼 수 있음
        - ex) service A 가 5분간 응답이 없으면, 관리자에게 slack DM 과 e-mail 보낸다.
- **Grafana**
    - `Prometheus` 와 항상 함께 연동되는 **시각화 툴**
    - `Prometheus` **자체 UI** 도 있고, API 제공을 하기에 **직접 시각화 대시보드를 구성** 가능
- **PromQL**
    - `Prometheus` 가 저장한 데이터 중 원하는 정보만 가져오기 위한 `Query Language`
    - `Time Series Data` 이며, `Multi-Dimensional Data` 이기에 처음 보면 다소 복잡하나,
    → `Prometheus 및 Grafana` 를 잘 사용하기 위해서 어느 정도 익혀야 함.
    

### 단점

- Scalability, High Availability
    - `Prometheus Server` 가 `Single node 로 운영`되어야 하기에 발생하는 문제
        
        ⇒ `Thanos` 라는 오픈소스를 활용해 `multi prometheus server` 운영
        

---

### Grafana

- `InfluxDB, Prometheus` 같은 `TimeSeriesDB 전용 시각화 툴`로 개발, 이후 `MySQL, PostgreSQL 같은 RDB`도 지원
- 현재는  Grafana Labs 회사에서 관리, Grafana외에도 Grafana Cloud, Grafana Enterprise 제품 존재
    - 상용 서비스는 추가 기능을 제공하는 것뿐 아니라 설치 및 운영 등의 기술 지원까지 포함
- playground 페이지도 제공해 Grafana Dashboard 사용해볼 수 있음
- `Docker` 로 쉽게 설치 가능
- 다양한 `외부 플러그인` 존재
- 다양한 `Dashboard` 존재

### Grafana Dashboard 모범사례

- 수많은 Metric 중 모니터링해야할 대상을 정하고 어떤 방식으로 시각할 것인지는 정답이 없음
- 단, Google 에서 제시한 전통 SW 모니터링을 위한 4가지 황금 지표 존재
    - Latency
        - 사용자가 요청 후 응답을 받기까지 걸리는 시간
    - Traffic
        - 시스템이 처리해야 하는 총 부하
    - Errors
        - 사용자의 요청 중 실패한 비율
    - Saturation
        - 시스템 포화 상태
        
- ML 기반의 서비스를 모니터링할 때도 위 4가지 지표를 염두에 두고 대시보드 작성하는 것이 좋음

---

### 실습

- minikube 쓰는 경우

```python
minikube start --driver=docker --cpus='4' --memory='4g'
```

### kube-prometheus-stack Helm Repo 추가

- Prometheus, Grafana 등을 k8s에 쉽게 설치하고 사용할 수 있도록 패키징된 Helm 차트
    - 버전 : kube-prometheus-stack-19.0.2

```python
# helm repo 추가
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# helm repo update
helm repo update
```

### kube-prometheus-stack 설치

```python
# helm install [RELEASE_NAME] prometheus-community/kube-prometheus-stack

helm install prom-stack prometheus-community/kube-prometheus-stack
# 모든 values 는 default 로 생성
# https://github.com/prometheus-community/helm-charts/blob/main/charts/kube-prometheus-stack/values.yaml

# 정상설치확인
# 최초 설치 시 docker image pull 로 인해 수 분의 시간이 소요될 수 있음
kubectl get pod -w
```

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2010.png)

- 실무에서 admin password, storage class, resource, ingress 등의 value 를 수정한 뒤 적용하는 경우, charts 를 clone한 뒤, `values.yaml` 을 수정하여 git으로 환경별 히스토리 관리

### How to Use

- 포트포워딩
    - 새로운 터미널을 열어 포트포워딩
    - `Grafana` 서비스
        - `kubectl port-forward svc/prom-stack-grafana 9000:80`
    - `Prometheus` 서비스
        - `kubectl port-forward svc/prom-stack-kube-prometheus-prometheus 9091:9090`
- Prometheus UI Login
    - [Localhost:9091](http://Localhost:9091) 로 접속
    - 다양한 PromQL 사용 가능 (Autocomplete 제공)
        - `kube_pod_container_status_running`
            - running status 인 pod 출력
        - `container_memory_usage_bytes`
            - container 별 memory 사용 현황 출력
    - 다양한 AlertRule 이 Default 로 생성되어 있음
        - Expression 이 PromQL 을 기반으로 정의되어 있음
        - 해당 AlertRule 이 발생하면 어디로 어떤 message 를 보낼 것인지도 정의할 수 있음
            - message send 설정은 default 로는 설정하지 않은 상태
                - Alertmanager configuration 을 수정하여 설정할 수 있음
                [https://github.com/prometheus-community/helm-charts/blob/7c5771add4ef2e92f520158078f8ea842c626337/charts/kube-prometheus-stack/values.yaml#L167](https://github.com/prometheus-community/helm-charts/blob/7c5771add4ef2e92f520158078f8ea842c626337/charts/kube-prometheus-stack/values.yaml#L167)
                - `values.yaml`
                
                ![Untitled](1-4%20Model%20%20b8ae8/Untitled%2011.png)
                
            - 해당 부분 바꾸면 됨

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2012.png)

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2013.png)

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2014.png)

### 위쪽에 Alerts Rule 을 설정할 수 있음

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2015.png)

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2016.png)

- 여기서 `expr` 의 `PromQL` 을 통해 적용 가능

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2017.png)

### Grafana UI Login

- [localhost:9000](http://localhost:9000) 으로 접속
- 디폴트(default) 접속 정보
    - id : `admin` / pw : `prom-operator`
    - 이것 또한 values.yaml 에 적혀있음
    
    ![Untitled](1-4%20Model%20%20b8ae8/Untitled%2018.png)
    
    ```bash
    # 또는 아래방법으로 디코딩을 통해 알 수 있음
    
    kubectl get secret --namespace default prom-stack-grafana -o jsonpath="{.data.admin-user}" | base64 --decode ; echo
    
    kubectl get secret --namespace default prom-stack-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
    ```
    

![Untitled](1-4%20Model%20%20b8ae8/Untitled%2019.png)

- `Configuration` - `Data sources 탭` 클릭
    - `Prometheus` 가 default 로 등록되어 있음
        - `Prometheus` 와 통신하는 URL 은 `쿠버네티스 service` 의 `DNS` 로 세팅
            - `Grafana` 와 `Prometheus` **모두 쿠버네티스 내부에서 통신**
- `Dashboards - Manage(or browse)` 탭 클릭
    - 다양한 대시보드가 default 로 등록되어 있음
        - `Kubernetes/Compute Resources/Namespace(Pods)` 확인
        
        ![Untitled](1-4%20Model%20%20b8ae8/Untitled%2020.png)
        
    - Time Range 조절 가능
    - Panel 별 PromQL 구성 확인 가능
    - 우측 상단의 Add Panel 버튼
        - Panel 추가 및 수정 가능
    - 우측 상단의 Save dashboard 버튼
        - 생성한, 수정한 Dashboard 를 영구히 저장하고 공유 가능
            - Dashboards - Manage 탭
                - Upload JSON file
                - Import from grafana.com
    
- 원하는 dashboards 가져오는 방법
    
    왼 : `grafana dashboard site` , 오 : dashboard > import
    
    ![Untitled](1-4%20Model%20%20b8ae8/Untitled%2021.png)
    
    ![Untitled](1-4%20Model%20%20b8ae8/Untitled%2022.png)