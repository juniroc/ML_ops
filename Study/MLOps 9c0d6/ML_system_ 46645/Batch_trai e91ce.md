# Batch_training_pattern

- 정기적으로 모델을 학습할 경우
- Batch job으로 학습하며, 특정 기간 간격(scheduling, trigger)으로  job

---

가장 쉬운 방법으로는 **crontab** (**linux**)

- cloud service 가능

### Workflow

1. Retrieve data from DWH 
2. Preprocess data
3. Train
4. Evaluate
5. Build model to prediction server
6. Store the model and server, and record the evaluation

![Batch_trai%20e91ce/Untitled.png](Batch_trai%20e91ce/Untitled.png)

**Workflow** 

**1. data filtering(value range) 및 data cleansing**

**2~4. hyperparameter / tune - 성능 올리기**

**5~6. system issue/error process**

**장점**

- 모델 재학습 및 업데이트

**단점**

- 에러 핸들링이 필요
- 전체적으로 자동화하는데 어려움 존재