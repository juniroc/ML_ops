<!-- #region -->
### User : 개인 / 기업 / 기관

-----

### Summary : Vision_Transformer to NNI (AutoML) - Hyperparameter & WebUI

-----

### versioning :
```
- version 1.0
```

-----

### Function List :
* Yaml 파일을 통해 환경 변수 정의 및 Hyperparameter
    ex)
    - Learning_rate
    - optimizer
    - patch_size or patch_number
    - vt_dim
    - vt_depth 
    - vt_head 
    - mlp_dim 등
* search_space.json 파일을 통한 hyperparmeter
* dataset_path 정의를 통한 custom_data 학습
* 기존에 학습된 모델에 추가 학습(resume)
* 이미지 사이즈 조절
* nni-monitoring (WebUI)

-----

### Install :
* nni -> https://github.com/microsoft/nni
* Vision_transformer pytorch git clone -> https://github.com/lucidrains/vit-pytorch / pip install vit-pytorch

-----

### How To Use :
1. Yaml 파일에 config 및 search_Space_path 정의

```
authorName: nujnim
experimentName: ViT_to_nni
trialConcurrency: 2
maxExecDuration: 99999d
maxTrialNum: 100
trainingServicePlatform: local
searchSpacePath: search_net_ViT.json
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  codeDir: .
  command: python main_ViT.py --download-dir ./dataset_real/chest_xray -a vit --batch-size 16 --epochs 5 --request-from-nni --num-classes 2 --gpu 0 
  gpuNum: 1

```


2. Search_space.json 정의 (Hyperparameter)

```
{
  "gamma": {
    "_type": "uniform",
    "_value": [0.5, 0.9]
  },
  "image_size": {
    "_type": "choice",
    "_value": [300]
  },
  "patch-num": {
    "_type": "choice",
    "_value": [15, 10, 6, 5]
  },
  "vt_dim": {
    "_type": "choice",
    "_value": [256, 128, 64]
  },
  "vt_depth": {
    "_type": "choice",
    "_value": [6, 5, 4]
  },
  "vt_heads": {
    "_type": "choice",
    "_value": [12, 8, 6]
  },
  "mlp_dim": {
    "_type": "choice",
    "_value": [1000, 500, 100]
  }
}
```

![image1](https://github.com/juniroc/ML_ops/blob/main/nni/Vision_Transformer/md_image/json_image.png)


3. main_vit.py 정의
    1) Env Config
        * tuning_parameter argparser 객체에 '직접' 저장
    2) main()
        * search_space의 tuning_parameter argparser 객체에 저장
    3) main_worker()
        * custom_data 가져오기 및 학습
    4) Train()
        * train_data로 학습 및 결과 도출
    5) Validation()
        * validation_data로 평가

4. webui 로 모델 모니터링
![image1](https://github.com/juniroc/ML_ops/blob/main/nni/Vision_Transformer/md_image/image_1.png)
![image2](https://github.com/juniroc/ML_ops/blob/main/nni/Vision_Transformer/md_image/image_2.png)
![image3](https://github.com/juniroc/ML_ops/blob/main/nni/Vision_Transformer/md_image/image_3.png)
![image4](https://github.com/juniroc/ML_ops/blob/main/nni/Vision_Transformer/md_image/image_4.png)

-----

### Function/Class :
1. utils_ViT.py
    * save_checkpoint(state, is_best, filename='checkpoint.pth.tar')
        - checkpoint 중 가장 성능이 좋은 데이터 'model_best.pth.tar' 로 저장
    * accuracy(output, target, topk=(1,n))
        - 정확도 측정 1등부터 n 등까지
    * adjust_learning_rate
        - optimization 과정에서 learning_rate를 조정
        
2. vit.py : Vision_Transformer model
    * PreNorm()
        - 데이터 정규화
        
    * FeedForward()
        - 순전파 과정
        
    * Attention()
        - 어텐션 레이어
        - 헤드 갯수, 디멘션(layer width)
        - query, key, value 곱 (patch내의 위치에 따른 영향 성분곱) 
        
    * Transformer()
        - 인코더부분 정의
        
    * ViT()
        - patch_size, image_size 적합 여부 판단 및 생성
        - 위치 임베딩
        - classification token 생성
        - dropout layer

3. main_vit.py
    * Env Config
        - tuning_parameter argparser 객체에 '직접' 저장
    * main()
        - search_space의 tuning_parameter argparser 객체에 저장
    * main_worker()
        - custom_data 가져오기 및 학습
    * Train()
        - train_data로 학습 및 결과 도출
    * Validation()
        - validation_data로 평가
-----

### SW / HW arch :

-----

### Function Flow archi :

-----

### Version List :


- python_3.8.5
- nni_1.9
- pytorch_1.7.1
- Vision_Transformer (https://github.com/lucidrains/vit-pytorch)

<!-- #endregion -->

```python

```
