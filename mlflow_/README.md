### 0. mlflow ui open
  * 도커 컨테이너 환경이므로 `-h 0.0.0.0` 추가.

```
mlflow ui -h 0.0.0.0
```

`capture_0` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/0.JPG)


---

### 1. hpyer_param_runs 라는 이름으로 experiments_id 생성
  * hyper_param_runs 라는 이름으로 experiment가 생성


```
mlflow experiments create -n hyper_param_runs
```


`capture_1` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/1.JPG)


---

### 2. hyperparameter 및 training을 위한 python 파일 생성
  * search_hyperopt.py : 하이퍼파라미터 parent_run 설정
    
  * train.py : 각 학습 과정의 child_run 설정 




---

### 3. MLproject 를 통해 workflow 생성 \
`MLproject` file

```
name: HyperparameterSearch

conda_env: conda.yaml

entry_points:
  # train Keras DL model
  train:
    parameters:
      training_data: {type: string, default: "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"}
      epochs: {type: int, default: 32}
      batch_size: {type: int, default: 16}
      learning_rate: {type: float, default: 1e-1}
      momentum: {type: float, default: .0}
      seed: {type: int, default: 97531}
    command: "python train.py {training_data}
                                    --batch-size {batch_size}
                                    --epochs {epochs}
                                    --learning-rate {learning_rate}
                                    --momentum {momentum}"


  # Use Hyperopt to optimize hyperparams of the train entry_point.
  hyperopt:
    parameters:
      training_data: {type: string, default: "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"}
      max_runs: {type: int, default: 12}
      epochs: {type: int, default: 32}
      metric: {type: string, default: "rmse"}
      algo: {type: string, default: "tpe.suggest"}
      seed: {type: int, default: 97531}
    command: "python -O search_hyperopt.py {training_data}
                                                 --max-runs {max_runs}
                                                 --epochs {epochs}
                                                 --metric {metric}
                                                 --algo {algo}
                                                 --seed {seed}"

  main:
    parameters:
      training_data: {type: string, default: "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"}
    command: "python search_hyperopt.py {training_data}"
```



---

### 4. MLproject workflow 실행
  * 1에서 생성된 experiment-id `1` 를 입력해주고, MLproject 파일 위치입력 
```
mlflow run -e hyperopt --experiment-id 1 ~/.
```

`capture_2` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/2.JPG)

---

### 5. result
  * 가장 높은 성능을 지닌 experiment 출력

`capture_3` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/3.JPG)


`capture_4` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/4.JPG)

---

### 6. model_serving
  * experiment와 함께 저장된 model 을 서빙

```
mlflow models serve -m mlruns/1/f8a4e7ee19374484948c1f9b5c2de7b5/artifacts/model -p 1234
```

`capture_5` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/5.JPG)
---

### 7. Inference_by_served_model
  * bentoml과 비슷한 방법으로 REST 형식 요청
    
```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations
```

`capture_6` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/6.JPG)

**result**


`capture_7` \
![python_exec](https://github.com/juniroc/ML_ops/blob/main/mlflow_/capture/7.JPG)
