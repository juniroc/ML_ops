# Chapter 2. Create Model

## Creating Model

**모델 개발 플로우**

`데이터 분석, 취득` → `모델 선정 / 파라미터 정리` → `전처리` → `학습` → `평가` → `빌드` → `시스템 평가`

1. **데이터 분석, 취득**
- 수집 가능한 데이터를 모두 모아 정리 및 분석하는 단계
- **머신러닝으로 해당 과제가 해결이 가능한지 파악** 이 핵심

1. **모델 선정 및 파라미터 정리**
- 해결할 과제나 데이터에 따라 사용 가능한 모델과 맞는 전처리가 다양함
→ 모델에 따라 파라미터도 달라짐
- 평가 결과가 좋은 모델과 사용할 수 있는 모델은 별개로 보아야 함
ex) 연산량이 커 추론이 10초나 걸리는 모델은 정확도가 좋아도 쓸모가 없음

1. **전처리**
- 데이터 형식에 따라 다양한 전처리 방법이 존재
ex) 자연어, 이미지, 수치형(정형) 데이터

1. **학습**
- 처음부터 복잡한 모델로 학습하지 않도록 주의
ex) 처음부터 앙상블 학습이나 NN 모델을 이용하기 보다는 logistic Regression, Decision Tree 와 같은 간단한 모델을 우선적으로 시험

1. **평가**
- 평가용 데이터를 이용해 모델의 좋고 나쁜 정도를 판단
ex)
분류 모델 : Acc, precision, Recall, confusion matrix 등 이용
회귀 모델 : MSE, MAE 등 이용
- **사실 실제로 배포한 이후에 대한 평가는 정답이 없는 경우가 있어 사용자가 직접 평가할 수 있는 환경을 제공하는 것이 좋을 것**

1. **빌드**
- 모델을 추론 시스템에 포함시키는 과정을 빌드라 생각하면 됨.
- 학습은 GPU 환경에서 진행하는 경우가 많지만, 추론은 **대부분이 CPU 환경**에서 진행
→ GPU 를 이용하더라도 **학습 GPU 환경과는 별개**
- 응답 속도가 **중요한 요소**이므로, **퍼포먼스 테스트나 부하 테스트**를 진행하는 것이 좋음

1. **시스템 평가**
- 추론기로써의 평가 과정
- **안정성, 응답 속도, 접속 테스트 등 실제 시스템으로 가동시키기 위한 항목 포함**

1. **실제 모델 개발**
- 위 모든 과정은 일방통행이 아님
→ 필요에 따라 앞의 과정으로 돌아가 다시 진행할 수 있음.

---

## Project, Model, Versioning

1. **프로젝트, 모델, 버저닝 관리**

1-1. **프로젝트 지칭 이름 필요**
→ 정식 명칭을 붙이지 않으면 의사소통에 큰 불편 야기
ex) animal_cls (동물분류 프로젝트 인 경우), Professor A(A가 진행하는 프로젝트) 등

1-2. **모델 버젼 관리 방법 정해야 함**

- 파라미터 별로 어떤 모델이 좋은 결과를 냈는지 관리
- 학습에 사용한 데이터의 관리 등
- 모델 버전 관리 방법

ex) [ `project명` ]_[ `git commit의 short hash` ]_[ `실험번호` ]

- `git commit의 short hash` : git 의 고유 커밋 ID (6자리)
- `실험번호` : commit 중 시도한 실험 번호 (commit 한번에 여러 실험이 존재할 수 있음)
→ 실험번호를 연동해 데이터나 파라미터 관리하는 것이 효율적
- **학습에 사용된 데이터 관리는 압축 후 위와 같은 버전명을 붙여 스토리지에 저장
또는 (용량 등으로 스토리지 사용이 제한될 경우) 데이터를 가져오는 쿼리를 함께 관리**
- **파라미터에도 버전명을 붙여 Json 이나 Yaml 형식으로 데이터와 함께 저장
또는 파라미터를 DB에 직접 등록**

![Untitled](Chapter%202%20%2070d81/Untitled.png)

- 모델 관리 데이터베이스 테이블 설계

- 책에서는 PostgreSQL 에서 SQLAlchemy 라는 ORM 라이브러리 이용
→ DB 의 Table을 클래스 객체로 취급핳는 라이브러리

---

### **아래는 ORM 라이브러리 사용 예시**

- `models.py`
    
    [mlsdp/models.py at main · wikibook/mlsdp](https://github.com/wikibook/mlsdp/blob/main/chapter2_training/model_db/src/db/models.py)
    
    - 윗 깃 참고한다
    
    ```python
    from sqlalchemy import Column, DateTime, ForeignKey, String, Text
    from sqlalchemy.sql.functions import current_timestamp
    from sqlalchemy.types import JSON
    from src.db.database import Base
    
    class Project(Base):
        __tablename__ = "projects"
    
        project_id = Column(
            String(255),
            primary_key=True,
            comment="主キー",
        )
        project_name = Column(
            String(255),
            nullable=False,
            unique=True,
            comment="プロジェクト名",
        )
        description = Column(
            Text,
            nullable=True,
            comment="説明",
        )
        created_datetime = Column(
            DateTime(timezone=True),
            server_default=current_timestamp(),
            nullable=False,
        )
    
    class Model(Base):
        __tablename__ = "models"
    
        model_id = Column(
            String(255),
            primary_key=True,
            comment="主キー",
        )
        project_id = Column(
            String(255),
            ForeignKey("projects.project_id"),
            nullable=False,
            comment="外部キー",
        )
        model_name = Column(
            String(255),
            nullable=False,
            comment="モデル名",
        )
        description = Column(
            Text,
            nullable=True,
            comment="説明",
        )
        created_datetime = Column(
            DateTime(timezone=True),
            server_default=current_timestamp(),
            nullable=False,
        )
    
    class Experiment(Base):
        __tablename__ = "experiments"
    
        experiment_id = Column(
            String(255),
            primary_key=True,
            comment="主キー",
        )
        model_id = Column(
            String(255),
            ForeignKey("models.model_id"),
            nullable=False,
            comment="外部キー",
        )
        model_version_id = Column(
            String(255),
            nullable=False,
            comment="モデルの実験バージョンID",
        )
        parameters = Column(
            JSON,
            nullable=True,
            comment="学習パラメータ",
        )
        training_dataset = Column(
            Text,
            nullable=True,
            comment="学習データ",
        )
        validation_dataset = Column(
            Text,
            nullable=True,
            comment="評価データ",
        )
        test_dataset = Column(
            Text,
            nullable=True,
            comment="テストデータ",
        )
        evaluations = Column(
            JSON,
            nullable=True,
            comment="評価結果",
        )
        artifact_file_paths = Column(
            JSON,
            nullable=True,
            comment="モデルファイルのパス",
        )
        created_datetime = Column(
            DateTime(timezone=True),
            server_default=current_timestamp(),
            nullable=False,
        )
    
    ```
    

- `cruds.py`
    
    [mlsdp/cruds.py at main · wikibook/mlsdp](https://github.com/wikibook/mlsdp/blob/main/chapter2_training/model_db/src/db/cruds.py)
    
    - 테이블 데이터 등록 및 참조를 위한 SQL 쿼리용 함수
    
    ```python
    import uuid
    from typing import Dict, List, Optional
    
    from sqlalchemy.orm import Session
    from src.db import models, schemas
    
    def select_project_all(db: Session) -> List[schemas.Project]:
        return db.query(models.Project).all()
    
    def select_project_by_id(
        db: Session,
        project_id: str,
    ) -> schemas.Project:
        return db.query(models.Project).filter(models.Project.project_id == project_id).first()
    
    def select_project_by_name(
        db: Session,
        project_name: str,
    ) -> schemas.Project:
        return db.query(models.Project).filter(models.Project.project_name == project_name).first()
    
    def add_project(
        db: Session,
        project_name: str,
        description: Optional[str] = None,
        commit: bool = True,
    ) -> schemas.Project:
        exists = select_project_by_name(
            db=db,
            project_name=project_name,
        )
        if exists:
            return exists
        else:
            project_id = str(uuid.uuid4())[:6]
            data = models.Project(
                project_id=project_id,
                project_name=project_name,
                description=description,
            )
            db.add(data)
            if commit:
                db.commit()
                db.refresh(data)
            return data
    
    def select_model_all(db: Session) -> List[schemas.Model]:
        return db.query(models.Model).all()
    
    def select_model_by_id(
        db: Session,
        model_id: str,
    ) -> schemas.Model:
        return db.query(models.Model).filter(models.Model.model_id == model_id).first()
    
    def select_model_by_project_id(
        db: Session,
        project_id: str,
    ) -> List[schemas.Model]:
        return db.query(models.Model).filter(models.Model.project_id == project_id).all()
    
    def select_model_by_project_name(
        db: Session,
        project_name: str,
    ) -> List[schemas.Model]:
        project = select_project_by_name(
            db=db,
            project_name=project_name,
        )
        return db.query(models.Model).filter(models.Model.project_id == project.project_id).all()
    
    def select_model_by_name(
        db: Session,
        model_name: str,
    ) -> List[schemas.Model]:
        return db.query(models.Model).filter(models.Model.model_name == model_name).all()
    
    def add_model(
        db: Session,
        project_id: str,
        model_name: str,
        description: Optional[str] = None,
        commit: bool = True,
    ) -> schemas.Model:
        models_in_project = select_model_by_project_id(
            db=db,
            project_id=project_id,
        )
        for model in models_in_project:
            if model.model_name == model_name:
                return model
        model_id = str(uuid.uuid4())[:6]
        data = models.Model(
            model_id=model_id,
            project_id=project_id,
            model_name=model_name,
            description=description,
        )
        db.add(data)
        if commit:
            db.commit()
            db.refresh(data)
        return data
    
    def select_experiment_all(db: Session) -> List[schemas.Experiment]:
        return db.query(models.Experiment).all()
    
    def select_experiment_by_id(
        db: Session,
        experiment_id: str,
    ) -> schemas.Experiment:
        return db.query(models.Experiment).filter(models.Experiment.experiment_id == experiment_id).first()
    
    def select_experiment_by_model_version_id(
        db: Session,
        model_version_id: str,
    ) -> schemas.Experiment:
        return db.query(models.Experiment).filter(models.Experiment.model_version_id == model_version_id).first()
    
    def select_experiment_by_model_id(
        db: Session,
        model_id: str,
    ) -> List[schemas.Experiment]:
        return db.query(models.Experiment).filter(models.Experiment.model_id == model_id).all()
    
    def select_experiment_by_project_id(
        db: Session,
        project_id: str,
    ) -> List[schemas.Experiment]:
        return (
            db.query(models.Experiment, models.Model)
            .filter(models.Model.project_id == project_id)
            .filter(models.Experiment.model_id == models.Model.model_id)
            .all()
        )
    
    def add_experiment(
        db: Session,
        model_version_id: str,
        model_id: str,
        parameters: Optional[Dict] = None,
        training_dataset: Optional[str] = None,
        validation_dataset: Optional[str] = None,
        test_dataset: Optional[str] = None,
        evaluations: Optional[Dict] = None,
        artifact_file_paths: Optional[Dict] = None,
        commit: bool = True,
    ) -> schemas.Experiment:
        experiment_id = str(uuid.uuid4())[:6]
        data = models.Experiment(
            experiment_id=experiment_id,
            model_version_id=model_version_id,
            model_id=model_id,
            parameters=parameters,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            evaluations=evaluations,
            artifact_file_paths=artifact_file_paths,
        )
        db.add(data)
        if commit:
            db.commit()
            db.refresh(data)
        return data
    
    def update_experiment_evaluation(
        db: Session,
        experiment_id: str,
        evaluations: Dict,
    ) -> schemas.Experiment:
        data = select_experiment_by_id(
            db=db,
            experiment_id=experiment_id,
        )
        if data.evaluations is None:
            data.evaluations = evaluations
        else:
            for k, v in evaluations.items():
                data.evaluations[k] = v
        db.commit()
        db.refresh(data)
        return data
    
    def update_experiment_artifact_file_paths(
        db: Session,
        experiment_id: str,
        artifact_file_paths: Dict,
    ) -> schemas.Experiment:
        data = select_experiment_by_id(
            db=db,
            experiment_id=experiment_id,
        )
        if data.artifact_file_paths is None:
            data.artifact_file_paths = artifact_file_paths
        else:
            for k, v in artifact_file_paths.items():
                data.artifact_file_paths[k] = v
        db.commit()
        db.refresh(data)
        return data
    ```
    

- 이때 데이터 조작을 실시하는 API 는 **FastAPI** 로 준비
    - `FastAPI` 의 엔드포인트는 파이썬 함수로 정의
    
    ex)
    
    ```python
    @router.get("/projects/all")
    def project_all(db: Session = Depends(get_db)):
        return cruds.select_project_all(db=db)
    ```
    
    - 위와 같이 정의하면 FastAPI 는 
    `http://<url>/projects/all` 이라는 API 엔드포인트를 공개

- `api.py`
    
    ```python
    from fastapi import APIRouter, Depends
    from sqlalchemy.orm import Session
    from src.db import cruds, schemas
    from src.db.database import get_db
    
    router = APIRouter()
    
    @router.get("/projects/all")
    def project_all(db: Session = Depends(get_db)):
        return cruds.select_project_all(db=db)
    
    @router.get("/projects/id/{project_id}")
    def project_by_id(
        project_id: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_project_by_id(
            db=db,
            project_id=project_id,
        )
    
    @router.get("/projects/name/{project_name}")
    def project_by_name(
        project_name: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_project_by_name(
            db=db,
            project_name=project_name,
        )
    
    @router.post("/projects")
    def add_project(
        project: schemas.ProjectCreate,
        db: Session = Depends(get_db),
    ):
        return cruds.add_project(
            db=db,
            project_name=project.project_name,
            description=project.description,
            commit=True,
        )
    
    @router.get("/models/all")
    def model_all(db: Session = Depends(get_db)):
        return cruds.select_model_all(db=db)
    
    @router.get("/models/id/{model_id}")
    def model_by_id(
        model_id: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_model_by_id(
            db=db,
            model_id=model_id,
        )
    
    @router.get("/models/project-id/{project_id}")
    def model_by_project_id(
        project_id: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_model_by_project_id(
            db=db,
            project_id=project_id,
        )
    
    @router.get("/models/name/{model_name}")
    def model_by_name(
        model_name: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_model_by_name(
            db=db,
            model_name=model_name,
        )
    
    @router.get("/models/project-name/{model_name}")
    def model_by_project_name(
        project_name: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_model_by_project_name(
            db=db,
            project_name=project_name,
        )
    
    @router.post("/models")
    def add_model(
        model: schemas.ModelCreate,
        db: Session = Depends(get_db),
    ):
        return cruds.add_model(
            db=db,
            project_id=model.project_id,
            model_name=model.model_name,
            description=model.description,
            commit=True,
        )
    
    @router.get("/experiments/all")
    def experiment_all(db: Session = Depends(get_db)):
        return cruds.select_experiment_all(db=db)
    
    @router.get("/experiments/id/{experiment_id}")
    def experiment_by_id(
        experiment_id: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_experiment_by_id(
            db=db,
            experiment_id=experiment_id,
        )
    
    @router.get("/experiments/model-version-id/{model_version_id}")
    def experiment_by_model_version_id(
        model_version_id: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_experiment_by_model_version_id(
            db=db,
            model_version_id=model_version_id,
        )
    
    @router.get("/experiments/model-id/{model_id}")
    def experiment_by_model_id(
        model_id: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_experiment_by_model_id(
            db=db,
            model_id=model_id,
        )
    
    @router.get("/experiments/project-id/{project_id}")
    def experiment_by_project_id(
        project_id: str,
        db: Session = Depends(get_db),
    ):
        return cruds.select_experiment_by_project_id(
            db=db,
            project_id=project_id,
        )
    
    @router.post("/experiments")
    def add_experiment(
        experiment: schemas.ExperimentCreate,
        db: Session = Depends(get_db),
    ):
        return cruds.add_experiment(
            db=db,
            model_version_id=experiment.model_version_id,
            model_id=experiment.model_id,
            parameters=experiment.parameters,
            training_dataset=experiment.training_dataset,
            validation_dataset=experiment.validation_dataset,
            test_dataset=experiment.test_dataset,
            evaluations=experiment.evaluations,
            artifact_file_paths=experiment.artifact_file_paths,
            commit=True,
        )
    
    @router.post("/experiments/evaluations/{experiment_id}")
    def update_evaluations(
        experiment_id: str,
        evaluations: schemas.ExperimentEvaluations,
        db: Session = Depends(get_db),
    ):
        return cruds.update_experiment_evaluation(
            db=db,
            experiment_id=experiment_id,
            evaluations=evaluations.evaluations,
        )
    
    @router.post("/experiments/artifact-file-paths/{experiment_id}")
    def update_artifact_file_paths(
        experiment_id: str,
        artifact_file_paths: schemas.ExperimentArtifactFilePaths,
        db: Session = Depends(get_db),
    ):
        return cruds.update_experiment_artifact_file_paths(
            db=db,
            experiment_id=experiment_id,
            artifact_file_paths=artifact_file_paths.artifact_file_paths,
        )
    ```
    

- FastAPI 는 Uvicorn 이라는 비동기 처리가 가능한 프레임워크에서 동작하는 라이브러리로 구성
- `Uvicorn` : 비동기 싱글 프로세스로 동작 ( ASGI 라는 표준 인터페이스 제공하는 프레임워크)
→ `Gunicorn` 에서 기동함으로써 멀티 프로세스로 사용 가능
- `Gunicorn` : 동기적 애플리케이션 인터페이스 제공 (WSGI 라는 표준 인터페이스 제공)
- 즉, Uvicorn 을 Gunicorn 에서 기동함으로써 비동기 처리와 멀티 프로세스 조합 가능

- `docker-compose.yaml`
    
    ```python
    version: "3"
    
    services:
      postgres:
        image: postgres:13.3
        container_name: postgres
        ports:
          - 5432:5432
        volumes:
          - ./postgres/init:/docker-entrypoint-initdb.d
        environment:
          - POSTGRES_USER=user
          - POSTGRES_PASSWORD=password
          - POSTGRES_DB=model_db
          - POSTGRES_INITDB_ARGS="--encoding=UTF-8"
        hostname: postgres
        restart: always
        stdin_open: true
    
      model_db:
        container_name: model_db
        image: shibui/ml-system-in-actions:model_db_0.0.1
        restart: always
        environment:
          - POSTGRES_SERVER=postgres
          - POSTGRES_PORT=5432
          - POSTGRES_USER=user
          - POSTGRES_PASSWORD=password
          - POSTGRES_DB=model_db
          - WORKERS=2
        entrypoint: ["./run.sh"]
        ports:
          - "8000:8000"
        depends_on:
          - postgres
    ```
    

---

### MLFlow 같은 Pipeline의 이점

- 작업 소모하는 자원이나 라이브러리 선정이 유연해짐
- 에러가 발생한 부분을 작업별로 분리하기 쉬움
- 워크로드 및 데이터에 따른 유연한 작업 관리 가능

**검토사항**

- ML/DL 에서는 컴퓨팅 리소스가 중요함.
- 전처리, 학습, 평가, 빌드, 시스템 평가 마다 개별 작업을 기동함과 동시에 자원을 확보
    - 개별 작업이 완료될 때마다 자원을 반환하는 것이 좋음
- 학습을 위해 항상 GPU 서버를 켜 둘 피료 없음

---

### 배치 학습 패턴

- 모델은 시간이 지나면 최신 데이터의 경향을 반영하지 못하는 등의 이유로 성능이 저하됨
→ 이를 해결하기 위해 축적된 데이터로 재학습해 **최신 데이터의 경향을 반영한 모델 생성**
- **즉, 모델을 정기적으로 갱신하고 싶은 경우 배치학습 패턴이 유용**

![Untitled](Chapter%202%20%2070d81/Untitled%201.png)

- 만약 mlflow 를 통해 학습하는 파이프라인을 생성해 놓았을 경우
→ **스케줄링 시스템(cron 등)**이나 작업 관리 서버에 **작업 실행 조건(일시, 데이터양, 이용량 등)을 등록하고 실행**
    
    ex)
    
    `~/my_dir/run_train.sh`
    
    ```bash
    #!/bin/bash
    
    set -eu
    
    mlflow run .
    ```
    
    - 위와 같은 `sh` 파일을 만들고 주기적으로 실행하는 커맨드 입력
    
    ```bash
    * 0 * * * cd /~/my_dir; ./run_train.sh
    
    # 매일 0 시에 학습
    ```
    

- 항상 최신으로 유지할 필요가 있는 경우, 에러가 발생하면 재시도하거나 운용자에게 통보해야 함.
→ 항상 최신일 필요가 없다면 에러만 통보 후 나중에 수동으로 재실행도 가능

### 검토사항

1. DWH 등에서 데이터 수집
2. 데이터 전처리
3. 학습
4. 평가
5. 모델 및 추론기의 빌드
6. 모델, 추론기, 평가의 기록