# Chapter 2-5. Feast 와 MLFlow 활용한 ML Project

### **MLFlow 와 Feast 를 이용한 ElasticNet 훈련 코드 생성**

- 사전 작업
    - feature_repo/example.py 에 아래 코드 추가
        
        ```python
        from feast import FeatureService
        driver_fs = FeatureService(name="driver_ranking_fv_svc",
                                   features=[driver_hourly_stats_view],
                                   tags={"description": "Used for training an ElasticNet model"})
        ```
        
    - Terminal 에서 아래 실행
        
        ```bash
        cd /home/jovyan/feature_repo
        rm -rf .ipynb_checkpoints/
        feast apply
        ```
        
- MLFlow 적용 전 모델 학습 코드 작성 해보기
    - `Jupyter Server` 에 새 폴더 (`/home/jovyan/jupyter/ml_project_1`) 생성하여 작업
    - `/home/jovyan/jupyter/ml_project_1/data` 생성
    - 학습 데이터 추출을 위한 `Query data` 생성 (`driver_orders.csv`)
        
        ```python
        import numpy as np
        import pandas as pd
        
        minio_uri = "http://172.17.0.3:9000"
        bucket_name = "feast-example"
        fname = "driver_stats.parquet"
        store_data = pd.read_parquet(f"{minio_uri}/{bucket_name}/{fname}")
        query_data = store_data.sample(n=10)[["event_timestamp","driver_id"]]
        query_data['trip_completed'] = np.random.randint(0, 2, query_data.shape[0])
        query_data.to_csv('/home/jovyan/jupyter/ml_project_1/data/driver_orders.csv', sep="\t", index=False)
        ```
        
    - `DriverRankingTrainModel` 클래스 생성
        
        ```python
        import pandas as pd
        from pprint import pprint
        from sklearn.linear_model import ElasticNet
        import feast
        
        class DriverRankingTrainModel:
            def __init__(self, repo_path: str, f_service_name: str, tuning_params={}) -> None:
                self._repo_path = repo_path
                self._params = tuning_params
                self._feature_service_name = f_service_name
        
            def get_training_data(self) -> pd.DataFrame:
                orders = pd.read_csv("/home/jovyan/jupyter/ml_project_1/data/driver_orders.csv", sep="\t")
                orders["event_timestamp"] = pd.to_datetime(orders["event_timestamp"])
        
                store = feast.FeatureStore(repo_path=self._repo_path)
                feature_service = store.get_feature_service(self._feature_service_name )
        
                training_df = store.get_historical_features(
                    entity_df=orders,
                    features=feature_service
                ).to_df()
                
                return training_df
            
            def train_model(self) -> str:
                model = ElasticNet(**self._params)
                target = "trip_completed"
                training_df = self.get_training_data()
                train_X = training_df[training_df.columns.drop(target).drop("event_timestamp")]
                train_y = training_df.loc[:, target]
        
                model.fit(train_X[sorted(train_X)], train_y)
                return model.coef_
        ```
        
        - 전역 변수 : _repo_path / _params / _feature_service_name
        - get_training_data
            - Feast FeatureStore 객체 생성
            - get_feature_service 로 서비스 객체 생성
                
                <aside>
                ✅ feast feature-services list 활용
                
                </aside>
                
            - get_historical_features 에 Query data 와 서비스를 기준으로 훈련 데이터 추출
        - train_model
            - ElasticNet 모델 생성
            - 학습/추론 데이터 분할
            - 모델 fitting
    - 실행 메인코드 작성
        
        ```python
        if __name__ == '__main__':
            REPO_PATH = "/home/jovyan/feature_repo"
            FEATURE_SERVICE_NAME = "driver_ranking_fv_svc"
            params_list = [{"alpha": 0.5, "l1_ratio": 0.15},
                           {"alpha": 0.75, "l1_ratio": 0.25},
                           {"alpha": 1.0, "l1_ratio": 0.5}]
        
            for params in params_list:
                model_cls = DriverRankingTrainModel(REPO_PATH, FEATURE_SERVICE_NAME, params)
                model_coef_ = model_cls.train_model()
                pprint(f"ElasticNet params: {params}")
                print(f"Model coefficients: {model_coef_}")
        ```
        
- MLFlow 적용하여 모델 학습 코드 수정 해보기
    - import mlflow 추가
    - [set_tracking_uri](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri), [sklearn.autolog](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html) 추가
        
        ```python
        def train_model(self) -> str:
            **mlflow.set_tracking_uri("sqlite:///mlruns.db")
            mlflow.sklearn.autolog()**
            
            model = ElasticNet(**self._params)
            target = "trip_completed"
            training_df = self.get_training_data()
            train_X = training_df[training_df.columns.drop(target).drop("event_timestamp")]
            train_y = training_df.loc[:, target]
        
            model.fit(train_X[sorted(train_X)], train_y)
            return model.coef_
        ```
        
    - [start_run](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run), [log_dict](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_dict), [log_model](https://www.mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry) 추가
        
        ```python
        **with mlflow.start_run() as run:**
        		model.fit(train_X[sorted(train_X)], train_y)
            **mlflow.log_dict({"features": ["driver_hourly_stats:conv_rate",
                                          "driver_hourly_stats:acc_rate",
                                          "driver_hourly_stats:avg_daily_trips"],
                             "feast_feature_service": self._feature_service_name,
                             "feast_feature_data": "driver_hourly_stats"}, "feast_data.json")
        		mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="sklearn-model",
                            registered_model_name="sk-learn-elasticnet-model"
                    )
        return {run.info.run_id}
        
        ...
        
        print(f"Model run id: {run_id}")**
        ```
        

### **MLFlow 와 Feast 를 이용한 예측 코드 생성**

- MLFlow 저장된 모델로 예측하는 코드 작성 해보기
    - 저장된 모든 모델 검색 예제
        
        ```python
        import mlflow
        from mlflow.tracking import MlflowClient
        from pprint import pprint
        
        client = MlflowClient()
        for rm in client.list_registered_models():
            pprint(dict(rm), indent=4)
        ```
        
    - 특정 이름의 모델 검색 예제
        
        ```python
        import mlflow
        from mlflow.tracking import MlflowClient
        from pprint import pprint
        
        client = MlflowClient()
        for mv in client.search_model_versions("name='sk-learn-**elasticnet**-model'"):
            pprint(dict(mv), indent=4)
        ```
        
    - 모델 불러오기 예제
        
        ```python
        import mlflow
        model_name = "sk-learn-**elasticnet**-model"
        model_version = "1"
        m_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(m_uri)
        print(model)
        ```
        
    - DriverRankingPredictModel 클래스 생성
        
        ```python
        class DriverRankingPredictModel:
            def __init__(self, repo_path:str, m_uri:str, feature_service_name:str) -> None:
                self._model = mlflow.sklearn.load_model(m_uri)
                self._fs = feast.FeatureStore(repo_path=repo_path)
                self._fsvc = self._fs.get_feature_service(feature_service_name)
        
            def __call__(self, entity_df): # 호출 가능한 객체로 만들어주는 방법
                    return self._predict(entity_df)
        
            def _predict(self, driver_ids):
                driver_features = self._fs.get_online_features(
                    entity_rows=[{"driver_id": driver_id} for driver_id in driver_ids],
                    features=self._fsvc
                )
                df = pd.DataFrame.from_dict(driver_features.to_dict())
                df["prediction"] = self._model.predict(df[sorted(df)])
                best_driver_id = df["driver_id"].iloc[df["prediction"].argmax()]
        
                return best_driver_id
        ```
        
    - 실행 메인코드 작성
        
        ```python
        if __name__ == "__main__":
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            # Change to your location
            REPO_PATH = "/home/jovyan/feature_repo"
            FEATURE_SERVICE_NAME = "driver_ranking_fv_svc"
            model_uri = "models:/sk-learn-elasticnet-model/3"
        
            model = DriverRankingPredictModel(REPO_PATH, model_uri, FEATURE_SERVICE_NAME)
            drivers = [1001, 1002, 1003]
            best_driver = model(drivers)
            print(f" Best predicted driver for completed trips: {best_driver}")
        ```
        
- 모델 Stage 변경하기
    
    ```python
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.transition_model_version_stage(
        name="sk-learn-elasticnet-model",
        version=3,
        stage="Production"
    )
    ```