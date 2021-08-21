### User : 개인 / 기업 / 기관

-----

### Summary : EfficientNet to NNI (AutoML) - Hyperparameter & WebUI

-----

### Versioning :

```
- version 1.0
```

-----

### Function List :

- Yaml 파일을 통해 환경 변수 정의 및 Hyperparameter ex)
    - download-dir 데이터 경로
    - worker_num 워커수
    - batch_size 배치 사이즈
    - base_resolution 초기 이미지 사이즈
    - num_classes 클래스 개수
    - gpu gpu 유무
    - resume pretrained 모델 유무
- search_space.json 파일을 통한 hyperparmeter
- dataset_path 정의를 통한 custom_data 학습
- 기존에 학습된 모델에 추가 학습(resume)
- nni-monitoring (WebUI)



-----

### Install :

- NNI -> https://github.com/microsoft/nni
- EfficientNet : NNI example에 존재하는 Efficientnet 이용

-----

### How To Use :
1. Yaml 파일에 Config 및 Search_Space_path 정의

```
authorName: nujnim
experimentName: example_efficient_net
trialConcurrency: 2
maxExecDuration: 99999d
maxTrialNum: 100
trainingServicePlatform: local
searchSpacePath: search_net-EfficientNet.json
useAnnotation: false
tuner:
  codeDir: .
  classFileName: tuner-EfficientNet.py
  className: FixedProductTuner
  classArgs:
    product: 2
trial:
  codeDir: .
  command: python main_EfficientNet.py custom --download-dir ./dataset_real -j 0 -a efficientnet --batch-size 16 --lr 0.048 --wd 1e-5 --epochs 5 --request-from-nni --resume ./model_best.pth.tar --resolution 300 --num-classes 2 --gpu 0
  gpuNum: 1
```

2. Search_space.json 정의 (Hyperparameter)

```
{
  "alpha": {
    "_type": "quniform",
    "_value": [1.0, 2.0, 0.05]
  },
  "beta": {
    "_type": "quniform",
    "_value": [1.0, 1.5, 0.05]
  },
  "gamma": {
    "_type": "quniform",
    "_value": [1.0, 1.5, 0.05]
  }
}
```
![image](https://github.com/juniroc/NNI_/blob/main/Efficient_Net/md_images/json_image.PNG)

3. main_EfficientNet.py 정의
    1) Env_config
        - tuning_parameter argparser 객체에 '직접' 저장
    
    2) main()
        - Search_Space의 Tuning_parameter Argparser 객체에 저장
    
    3) main_worker()
        - Custom_data 
        - pretrained EfficinetNet 이용 : 모델을 불러와 추가로 Fine_Tuning
        - 이용하지 않은 경우 : Hyperparameter를 통해 원하는 구조 model 생성
    
    4) Train()
        - Train_data로 학습 및 결과 도출
        
    5) Validation()
        - Validation_data로 평가

4. webui로 모델 모니터링
    
    ![image](https://github.com/juniroc/NNI_/blob/main/Efficient_Net/md_images/image1.png)
    ![image](https://github.com/juniroc/NNI_/blob/main/Efficient_Net/md_images/image2.png)
    ![image](https://github.com/juniroc/NNI_/blob/main/Efficient_Net/md_images/image3.png)
    

-----

### Function / Class :

1. utils.py
    - save_checkpoint(state, is_best, filename='checkpoint.pth.tar')
        - checkpoint 중 가장 성능이 좋은 데이터 'model_best.pth.tar' 로 저장
    
    - accuracy(output, target, topk=(1,n))
        - 정확도 측정 1등부터 n 등까지
    
    - adjust_learning_rate
        - optimization 과정에서 learning_rate를 조정
    
2. efficientnet_pytorch/model.py
    - MBConvBlock()
        - Mobile Inverted Resiudal Bottlenect Block 레이어 정의
    
    - EfficientNet()
        - MBConvBlock layer를 기반 레이어 구축
        - from_pretrained : 데이터 구조와 가중치가 함께 저장
        - from_name : 데이터 가중치 파라미터만 저장
        - get_image_size : 이미지 사이즈 출력
        - check_model_name_is_valid : 모델 적절성(존재) 유무 판단

3. efficientnet_pytorch/utils.py
    - config
        - relu_fn() : multiply sigmoid function 
        - round_filters() : 변경된 이미지 구조 넓이에 맞도록 filter 조정
        - round_repeats() : 레이어 깊이 만큼 반복
        - drop_connect() : (dropout과 비슷한 원리) 랜덤하게 끊기
    
    - get_same_padding_conv2d()
        - 기존과 같은 사이즈가 되도록 패딩
    
    - Conv2dDynamicSamePadding()
        - 원하는 이미지 사이즈가 정해진 경우 그 이미지 사이즈로 패딩 

    - Conv2dStaticSamePadding()
        - base 이미지 그대로 유지하기 위한 패딩
    
    - efficientnet_params()
        - pretrained model 파라미터 추출
    
    - efficientnet()
        - 기존 모델 구조 파라미터
    
    - get_model_params()
        - 모델 파라미터 가져오기
    
    - load_pretrained_weights()
        - 사전학습된 가중치 로드

4. main_EfficientNet.py
    - Env Config
        - tuning_parameter argparser 객체에 '직접' 저장

    - main()
        - search_space parameter들 parser 객체에 넣기
    
    - main_worker()
        - custom_data 가져오기
    
    - Train()
        - train_data로 학습 및 결과 도출
        
    - Validation()
        - validation_data로 평가
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
