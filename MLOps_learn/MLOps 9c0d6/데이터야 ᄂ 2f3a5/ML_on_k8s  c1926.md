# ML_on_k8s (야놀자)

![Untitled](ML_on_k8s%20%20c1926/Untitled.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%201.png)

EMR (Elastic MapReduce)

: AWS에 제공해주는 완전제공형 **빅데이터 플랫폼**

- 하둡, Spark, Hive 등의 프레임워크 제공

MSK (Managed Streaming for Apache Kafka)

: Kafka를 이용해 스트리밍 데이터를 처리하는 애플리케이션을 빌드하고 실행할 수 있는 완전 관리형 서비스

![Untitled](ML_on_k8s%20%20c1926/Untitled%202.png)

RDS (Amazon Relational Database Service)

: 분산 관계형 데이터베이스 (from Amazon)

→ 애플리케이션 내에서 관계형 데이터베이스의 설정, 운영, 스케일링을 단순케 하도록 설계된 클라우드 내에서 동작하는 웹 서비스

- 데이터베이스 소프트웨어를 패치하거나 데이터베이스를 백업하거나 시점 복구를 활성화하는 것과 같은 복잡한 관리 프로세스들은 자동으로 관리됨

![Untitled](ML_on_k8s%20%20c1926/Untitled%203.png)

- 서비스 유형에 따라 EKS 분류

EKS (Elastic Kubernetes Service)

: Amazon의 관리형 쿠버네티스 서비스

![Untitled](ML_on_k8s%20%20c1926/Untitled%204.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%205.png)

 

![Untitled](ML_on_k8s%20%20c1926/Untitled%206.png)

- 용도에 따라 Mode 설정하여 이용 (Local / Client mode)
ex) spark만 이용시 Local mode / spark + ray 이용시 Client mode

![Untitled](ML_on_k8s%20%20c1926/Untitled%207.png)

- Spark에서 분산처리(전처리)한 데이터를 parquet으로 저장하여 Ray로 학습하는 프로세스 *(Uber)

![Untitled](ML_on_k8s%20%20c1926/Untitled%208.png)

- Mode 별 분류

![Untitled](ML_on_k8s%20%20c1926/Untitled%209.png)

- Executor 내 Shuffle(대량 데이터 사용으로 인해)로 인해 OOM 발생 가능성 있음 (??)
→ Executor Pod 마다 Volume Mount 가 필요함 (OOM 발생 예방)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2010.png)

- Executor Pod 할당에 시간이 지연될 수 있음
→ 이는 (Spark 옵션인) Spark Task 대기시간 조절을 통해 해결 가능

![Untitled](ML_on_k8s%20%20c1926/Untitled%2011.png)

- PySpark 와 Ray 같은 프레임워크를 섞어 쓰는 경우 python 영역에서 OOM 가능성
→ 전처리 후 Notebook 내에서 Spark Session을 내리거나, Python 라이브러리 전용 Dag 분리해 작업 (Pod 초기화 시간이 약간 소요)
- Off-heap 이란?
    
    ![Untitled](ML_on_k8s%20%20c1926/Untitled%2012.png)
    

![Untitled](ML_on_k8s%20%20c1926/Untitled%2013.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2014.png)

- Notebook 을 EFS에 저장

![Untitled](ML_on_k8s%20%20c1926/Untitled%2015.png)

- Papermill과 Airflow 를 통한 notebook 파일 스케쥴링
- 상황에 맞는 방법 이용하는게 좋을 것

![Untitled](ML_on_k8s%20%20c1926/Untitled%2016.png)

---

- Scala Spark 를 사용하는 경우
    
    ![Untitled](ML_on_k8s%20%20c1926/Untitled%2017.png)
    
    - Scala Spark의 경우
    
    ![Untitled](ML_on_k8s%20%20c1926/Untitled%2018.png)
    
    - Scala Spark를 이용하는 경우
    

---

![Untitled](ML_on_k8s%20%20c1926/Untitled%2019.png)

### 모델 학습 및 배포

![Untitled](ML_on_k8s%20%20c1926/Untitled%2020.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2021.png)

- ML-flow를 통해 파라미터와 성능 체크

![Untitled](ML_on_k8s%20%20c1926/Untitled%2022.png)

BentoML

- Annotation 기반으로 API를 쉽게 만들 수 있음
- Swagger UI 를 제공하여 테스트나 문서화가 편리
- K8S, KFServing, Kubeflow 등에 대한 직접적인 배포 지원하나, 팀 별 정책에 따른 컨테이너 이미지화하여 직접 배포가 나음

![Untitled](ML_on_k8s%20%20c1926/Untitled%2023.png)

왼쪽 : Model Artifact를 API에 임베딩하고, 다음 Wrapping API 호출

오른쪽 : 모델을 파일로 공유 스토리지에 떨구고, Wrapping API에서 모델 파일 서빙

![Untitled](ML_on_k8s%20%20c1926/Untitled%2024.png)

### 생각할 거리 (열린 질문)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2025.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2026.png)

---

### ETC

![Untitled](ML_on_k8s%20%20c1926/Untitled%2027.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2028.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2029.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2030.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2031.png)

![Untitled](ML_on_k8s%20%20c1926/Untitled%2032.png)