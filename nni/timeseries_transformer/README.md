### User : 개인 / 기업 / 기관

-----

### Summary : Transformer to NNI (AutoML) - Hyperparameter & WebUI

-----

### Versioning :

```
- version 1.0
```

-----

### Function List :

- Yaml 파일을 통해 환경 변수 정의 및 Hyperparameter ex)
    - Seed : random seed
    - gpu : gpu 유/무
    - resume pretrained 모델 유무
- search_space.json 파일을 통한 hyperparmeter
- dataset_path 정의를 통한 custom_data 학습
- 기존에 학습된 모델에 추가 학습(resume)
- nni-monitoring (WebUI)


<!-- #region -->
-----

### Install :

- NNI -> https://github.com/microsoft/nni
- EfficientNet : NNI example에 존재하는 Efficientnet 이용

-----

### How To Use :
1. Yaml 파일에 Config 및 Search_Space_path 정의

```
authorName: nujnim
experimentName: Transformer_to_nni
trialConcurrency: 2
maxExecDuration: 99999d
maxTrialNum: 100
trainingServicePlatform: local
searchSpacePath: search_transformer.json
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: minimize
trial:
  codeDir: .
  command: python Transformer_energy_nni.py --seed 42 --gpu 0 --request-from-nni 
  gpuNum: 1

```

2. Search_space.json 정의 (Hyperparameter)

```
{
  "batch_size": {
    "_type": "choice",
    "_value": [32, 64]
  },
  "time_steps": {
    "_type": "choice",
    "_value": [24, 30, 50, 100]
  },
  "num_head": {
    "_type": "choice",
    "_value": [8, 4, 2]
  },
  "embed_dim": {
    "_type": "choice",
    "_value": [256, 128, 64]
  },
  "num_encoderlayers": {
    "_type": "choice",
    "_value": [6, 4, 2]
  },
  "num_decoderlayers": {
    "_type": "choice",
    "_value": [6, 4, 2]
  },
  "dropout": {
    "_type": "choice",
    "_value": [0.2, 0.3, 0.4]
  }
}
```


![image](https://github.com/juniroc/ML_ops/blob/main/nni/timeseries_transformer/capture/json.PNG)
<!-- #endregion -->

3. energy_transformer_nni.py 정의
    1) Env_config
        - tuning_parameter argparser 객체에 '직접' 저장
    
    2) Dacon_preprocessing()
        - create_sequences()
        - get_each_id_set()
        - get_group()
        - 위 함수들을 이용해 윈도우 사이즈 별 
    
    3) main()
        - Search_Space의 Tuning_parameter Argparser 객체에 저장
    
    4) FFN() & Energy_Model()
        - Transformer 기반의 타임시리즈 모델 구축
    
    5) main_worker() 
        - 모델 파라미터 정의
        - loss, optimizer 정의 
        - 최적화된 모델 저장
    
    6) Train()
        - Train_data로 학습 및 결과 도출
        
    7) Validation()
        - Validation_data로 평가

4. webui로 모델 모니터링
    
    ![image](https://github.com/juniroc/ML_ops/blob/main/nni/timeseries_transformer/capture/webui_1.png)
    ![image](https://github.com/juniroc/ML_ops/blob/main/nni/timeseries_transformer/capture/webui_2.png)
    ![image](https://github.com/juniroc/ML_ops/blob/main/nni/timeseries_transformer/capture/webui_3.png)    

-----

### Function / Class :

1. utils.py
    - save_checkpoint(state, is_best, filename='checkpoint.pth.tar')
        - checkpoint 중 가장 성능이 좋은 데이터 'model_best.pth.tar' 로 저장
    
    - accuracy(output, target, topk=(1,n))
        - 정확도 측정 1등부터 n 등까지
    
    - adjust_learning_rate
        - optimization 과정에서 learning_rate를 조정
    
2. energy_transformer_nni.py
    - preprocessing_()
        - 데이터 전처리
        - 타임스텝 단위로 데이터 window 생성
    
    - Energy_Model
        - 각 데이터를 컬럼별로 embedding 시킴 (데이터의 의미 추출) 
            ex) num(건물번호)를 embedding 시켜 모델이 건물별로 구분하도록 학습(명목형 함수)
        - 위치 데이터 embedding
        - embedding 레이어를 통과한 데이터들을 enc, dec 에 들어가도록 합
        - transformer 모델에 enc, dec 를 넣어줌
        - 이후 나온 결과를 ffn 레이어를 통과
        - 다시한번 residual connection 진행
        - 결과 데이터 추출
    - FFN
        - 주어진 state_size (dense 레이어 노드 갯수)를 이용해 Linear layer 생성
        - Linear layers 통과할 때 residual connection 진행 식: h(x) = f(x) + x

    - Env Config
        - tuning_parameter argparser 객체에 '직접' 저장

    - main()
        - search_space parameter들 parser 객체에 넣기
    
    - main_worker()
        - hyperparameter 모델에 적용해 구조 생성
    
    - Train()
        - train_data로 학습 및 결과 도출
        
    - Validation()
        - validation_data로 평가

    - smape()
        - 평가 척도 
-----

### SW / HW arch :

-----

### Function Flow Archi :

-----

### Version List :

- Python_3.8.5

- nni_1.9

- pytorch_1.7.1


```python

```
