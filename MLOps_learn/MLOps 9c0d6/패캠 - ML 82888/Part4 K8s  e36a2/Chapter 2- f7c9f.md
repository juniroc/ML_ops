# Chapter 2-2. Feature Store (Feast - 실습)

![Untitled](Chapter%202-%20f7c9f/Untitled.png)

![Untitled](Chapter%202-%20f7c9f/Untitled%201.png)

![Untitled](Chapter%202-%20f7c9f/Untitled%202.png)

![Untitled](Chapter%202-%20f7c9f/Untitled%203.png)

![Untitled](Chapter%202-%20f7c9f/Untitled%204.png)

---

---

### **Feast Feature Store 를 생성하여 각 Feature 들을 정의하여 Store 에 배포하기**

- feature store 작업을 할 경로로 이동
    
    ```bash
    mkdir -p mlops/feature_store && cd mlops/feature_store
    ```
    
- Jupyter Lab Docker Container 실행
    
    ```bash
    docker run -d --name jupyter -p 8888:8888 -e JUPYTER_TOKEN='password' \
    -v "$PWD":/home/jovyan/jupyter --user root --restart=always \
    -it jupyter/base-notebook start.sh jupyter lab
    ```
    
- [localhost:8888](http://localhost:8888) 접속
- jupyter 폴더에서 새 노트북 파일 생성
- Store 생성과 배포
    - **Feast 설치하기**
        
        ```python
        %%sh
        pip install feast -U -q
        pip install Pygments -q
        ```
        
        - pip install ... -U : 지정된 모든 패키지를 최신으로 업그레이드
        - pip install ... -q : 출력 최소화
        - Pygments : 코드 강조 기능
    - Runtime 재시작
    - Feast 저장소 초기화
        
        ```python
        !feast init feature_repo
        ```
        
    - 생성된 Feature 저장소 확인
        
        ```python
        %cd feature_repo
        !ls -R
        ```
        
        - 각 폴더/파일 확인
        
        ```python
        !pygmentize -f terminal16m example.py
        ```
        
        - data : Feast 에서 제공하는 데모 데이터 (parquet 형식)
        - example.py : 데모 데이터의 Feature 정의
            
            ![Untitled](Chapter%202-%20f7c9f/Untitled%205.png)
            
            - [Data Source](https://rtd.feast.dev/en/master/#module-feast.data_source)
                
                ![Untitled](Chapter%202-%20f7c9f/Untitled%206.png)
                
            - [Feature](https://rtd.feast.dev/en/master/#module-feast.feature)
                - name, dtype, labels 정의
            - [Entity](https://rtd.feast.dev/en/master/#module-feast.entity)
                - 관련된 Feature 들의 모음으로 primary key 역할
                - Entity key
                    
                    ![Untitled](Chapter%202-%20f7c9f/Untitled%207.png)
                    
            - Feature View 내 [다른 데이터 원천](https://docs.feast.dev/reference/data-sources) 사용
                - BigQuery 예시
                    
                    ```python
                    driver_stats_fv = **FeatureView**(
                        name="driver_activity",
                        **entities**=["driver"],
                        features=[
                            **Feature**(name="trips_today", dtype=ValueType.INT64),
                            Feature(name="rating", dtype=ValueType.FLOAT),
                        ],
                        batch_source=**BigQuerySource**(
                            table_ref="feast-oss.demo_data.driver_activity"
                        )
                    )
                    ```
                    
                - 오프라인(학습) 데이터와 온라인(추론) 환경 모두에서 일관된 방식으로 Feature 데이터를 모델링 할 수 있게 함
                - 만약 FeatureView 가 특별한 entity 와 관계가 없는 feature 들만 포함한다면 entities 가 없이 (entities=[ ]) 구성될 수 있음
                    
                    ```python
                    global_stats_fv = FeatureView(
                        name="global_stats",
                        entities=[],
                        features=[
                            Feature(name="total_trips_today_by_all_drivers", dtype=ValueType.INT64),
                        ],
                        batch_source=BigQuerySource(
                            table_ref="feast-oss.demo_data.global_stats"
                        )
                    )
                    ```
                    
                - Entity aliasing
                    
                    ```python
                    # location_stats_feature_view.py
                    from feast import Entity, Feature, FeatureView, FileSource, ValueType
                    location = Entity(name="location", join_key="location_id", value_type=ValueType.INT64)
                    location_stats_fv= FeatureView(
                        name="location_stats",
                        entities=["location"],
                        features=[
                            Feature(name="temperature", dtype=ValueType.INT32)
                        ],
                        batch_source=BigQuerySource(
                            table_ref="feast-oss.demo_data.location_stats"
                        ),
                    )
                    
                    # temperatures_feature_service.py
                    from location_stats_feature_view import location_stats_fv
                    temperatures_fs = FeatureService(
                        name="temperatures",
                        features=[
                            location_stats_feature_view
                                .with_name("origin_stats")
                                .with_join_key_map(
                                    {"location_id": "origin_id"}
                                ),
                            location_stats_feature_view
                                .with_name("destination_stats")
                                .with_join_key_map(
                                    {"location_id": "destination_id"}
                                ),
                        ],
                    )
                    ```
                    
            
            - Feature service
                - 여러 개의 Feature View 를 포함할 수 있음
                - ML모델 당 하나를 생성하여 모델이 사용하는 Feature 를 추적할 수 있음
                    - 누적된 Feature 값들을 얻기 위한 Feature View 들을 통해 훈련 데이터를 생성해내는 데 사용할 수 있음.
                    - 단일 데이터셋은 여러 개의 Feature view 들로부터의 Feature 들로 구성 가능함
                    - Online Store 나 Offline Store 로부터 Feature 들을 추출할 때 여러 개의 Feature Views 로 구성된 Feature service 로부터 검색할 수 있음
                        
                        ```python
                        from feast import FeatureStore
                        # Store 초기화
                        feature_store = FeatureStore('.')
                        # Feature Serivce 추출
                        feature_service = feature_store.get_feature_service("driver_activity")
                        # 해당 Feature Service 에 해당하는 Feature 들을 entity_dict 기준으로 추출
                        # 1. Online Store
                        features = feature_store.get_online_features(
                            features=feature_service, entity_rows=[entity_dict]
                        )
                        # 2. Offline Store
                        
                        ```
                        
                        - Online Store
                            - latency 가 낮은 Online feature 조회에 사용됨
                            - materialize 명령을 사용하여 Feature View 의 데이터 원천으로부터 Online Store 로 load 됨
                            - Online Store 의 Feature 저장 방식은 데이터 원천의 저장 방식을 그대로 따름
                            - Online Store 와 데이터 원천 간의 가장 큰 차이점은 entity key 마다의 최신 Feature 값만 저장됨. 누적값들이 저장되지 않음
                            - 데이터 원천 예
                            
                            ![Untitled](Chapter%202-%20f7c9f/Untitled%208.png)
                            
                            - Online Store 모습
                            
                            ![Untitled](Chapter%202-%20f7c9f/Untitled%209.png)
                            
                        - Offline Store
                            - 누적되는 Feature 들로부터 훈련 데이터를 생성할 때 사용
                            - 빠른 추론을 위해 Feature 들이 필요할 때 Offline Store 로부터 Feature 들을 materializing 하여 load 함
                            
        - feature_store.yaml : registry, provider, online_store 경로 설정
            
            ```bash
            !pygmentize feature_store.yaml
            ```
            
            - registry
                - 모든 Feature 들의 정의와 메타데이터가 모여 있는 곳
                - 작업자들의 공동 작업을 가능하게 함
                - Feast 배포마다 하나의 registry 가 존재함
                - 파일 기반을 기본으로 하며 Local, S3, GCS 를 기반으로 할 수 있음
                - entities, feature views, feature services 등의 registry 구성 요소들은 apply 명령에 의해 업데이트되지만, 각 구성 요소들의 메타데이터는 materialization 과 같은 작업에 의해 업데이트 가능함
                - registry 내 모든 Feature View 들을 확인하고 싶은 경우
                
                ```python
                fs = FeatureStore("my_feature_repo/")
                print(fs.list_feature_views())
                ```
                
                - registry 내 특정 Feature View 를 확인하고 싶은 경우
                
                ```python
                fs = FeatureStore("my_feature_repo/")
                fv = fs.get_feature_view(“my_fv1”)
                ```
                
            - provider
                - provider 별로 구성 요소(Offline Store, Online Store, Infra, Computing) 활용
                    - 예) GCP : [BigQuery](https://cloud.google.com/bigquery) 를 Offline Store, [Datastore](https://cloud.google.com/datastore) 를 Online Store
                - 기본 Provider : Local / GCP / AWS
                - provider의 솔루션 외에 병행 사용 가능
                    - 예) GCP : DataStore 대신 Redis 를 Online Store 로 사용
                - provider 를 [사용자 정의](https://docs.feast.dev/how-to-guides/creating-a-custom-provider)하여 사용도 가능함
    - 원본 데이터 확인
        
        ```bash
        import pandas as pd
        pd.read_parquet("data/driver_stats.parquet")
        ```
        
    - 정의된 Feature 들을 적용(배포)하기
        - 적용 전에 feature_repo 에 있는 notebook 관련 파일 삭제 필요
        
        ```python
        !rm -rf .ipynb_checkpoints/
        !feast apply
        ```
        

---

### **Store 로부터 훈련 데이터 추출하기**

- 훈련 데이터 추출
    - get_historical_features 로 데이터 추출
        
        ```python
        from datetime import datetime, timedelta
        import pandas as pd
        
        from feast import FeatureStore
        
        # The entity dataframe is the dataframe we want to enrich with feature values
        entity_df = pd.DataFrame.from_dict(
            {
                "driver_id": [1001, 1002, 1003],
                "label_driver_reported_satisfaction": [1, 5, 3], 
                "event_timestamp": [
                    datetime.now() - timedelta(minutes=11),
                    datetime.now() - timedelta(minutes=36),
                    datetime.now() - timedelta(minutes=73),
                ],
            }
        )
        
        store = FeatureStore(repo_path=".")
        
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
        ).to_df()
        
        print("----- Feature schema -----\n")
        print(training_df.info())
        
        print()
        print("----- Example features -----\n")
        print(training_df.head())
        ```
        

![Untitled](Chapter%202-%20f7c9f/Untitled%2010.png)

`entity_df`

![Untitled](Chapter%202-%20f7c9f/Untitled%2011.png)

`training_df`

![Untitled](Chapter%202-%20f7c9f/Untitled%2012.png)

### **Online Store 로 데이터를 적재하고 추론을 위한 Feature Vector 가져오기**

- Online Store 로 데이터 적재
    - Serving 을 하기 위해 materialize-incremental 명령어를 사용하여 가장 최근 실행된 materialize 이후의 모든 새로운 feature 값들을 serialization 시켜 준다.
        
        ```python
        from datetime import datetime
        !feast materialize-incremental {datetime.now().isoformat()}
        ```
        
    
    ![Untitled](Chapter%202-%20f7c9f/Untitled%2013.png)
    
    - materialized features 확인
        
        ```python
        print("--- Data directory ---")
        !ls data
        
        import sqlite3
        import pandas as pd
        con = sqlite3.connect("data/online_store.db")
        print("\n--- Schema of online store ---")
        print(
            pd.read_sql_query(
                "SELECT * FROM feature_repo_driver_hourly_stats", con).columns.tolist())
        con.close()
        ```
        
    
    ![Untitled](Chapter%202-%20f7c9f/Untitled%2014.png)
    
- 추론을 위한 Feature Vector 가져오기
    - get_online_features 로 데이터 추출
        
        ```python
        from pprint import pprint
        from feast import FeatureStore
        
        store = FeatureStore(repo_path=".")
        
        feature_vector = store.get_online_features(
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
            entity_rows=[
                {"driver_id": 1004},
                {"driver_id": 1005},
            ],
        ).to_dict()
        
        pprint(feature_vector)
        ```
        
    
    ![Untitled](Chapter%202-%20f7c9f/Untitled%2015.png)