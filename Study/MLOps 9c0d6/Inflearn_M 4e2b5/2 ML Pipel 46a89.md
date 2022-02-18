# 2. ML Pipeline Steps

![2%20ML%20Pipel%2046a89/Untitled.png](2%20ML%20Pipel%2046a89/Untitled.png)

![2%20ML%20Pipel%2046a89/Untitled%201.png](2%20ML%20Pipel%2046a89/Untitled%201.png)

![2%20ML%20Pipel%2046a89/Untitled%202.png](2%20ML%20Pipel%2046a89/Untitled%202.png)

![2%20ML%20Pipel%2046a89/Untitled%203.png](2%20ML%20Pipel%2046a89/Untitled%203.png)

![2%20ML%20Pipel%2046a89/Untitled%204.png](2%20ML%20Pipel%2046a89/Untitled%204.png)

- versioning 이 중요 (Reproducebility : 재현 가능성이 떨어짐)
- data drift 확인 : 모델의 성능 저하를 야기 하는 모델 입력 데이터의 변경 내용

![2%20ML%20Pipel%2046a89/Untitled%205.png](2%20ML%20Pipel%2046a89/Untitled%205.png)

[Tensorflow Data Validation 사용하기](https://zzsza.github.io/mlops/2019/05/12/tensorflow-data-validation-basic/)

- TFDV를 사용하는 것이 제일 좋음
- TFDV - Data를 더 쉽게 이해하고 점검할 수 있도록 도와주는 라이브러리
→ 시각적으로  missing, abnormal, non-uniform value 등 데이터 점검 가능
- Data Validation은 정말 중요함

![2%20ML%20Pipel%2046a89/Untitled%206.png](2%20ML%20Pipel%2046a89/Untitled%206.png)

![2%20ML%20Pipel%2046a89/Untitled%207.png](2%20ML%20Pipel%2046a89/Untitled%207.png)

![2%20ML%20Pipel%2046a89/Untitled%208.png](2%20ML%20Pipel%2046a89/Untitled%208.png)

![2%20ML%20Pipel%2046a89/Untitled%209.png](2%20ML%20Pipel%2046a89/Untitled%209.png)

![2%20ML%20Pipel%2046a89/Untitled%2010.png](2%20ML%20Pipel%2046a89/Untitled%2010.png)

![2%20ML%20Pipel%2046a89/Untitled%2011.png](2%20ML%20Pipel%2046a89/Untitled%2011.png)

- GPU 가 얼만큼 있을 때, 그것을 Automl 을 어떻게 돌릴지 생각도 해야함.
→ automl을 아무리돌려도 성능이 고정되어있어 early stopping도 고려해야함
- 파이프라인의 꽃 == AutoML

![2%20ML%20Pipel%2046a89/Untitled%2012.png](2%20ML%20Pipel%2046a89/Untitled%2012.png)

- 다차원 적인 분석 (기존의 recall, auc, mse 를 넘어서)
- TFX에는 TFMA라는 것이 있는데 → slicing하는 것 (단, 손으로 일일이 해야함
→ ex) 연봉이 5만달러가 넘는지 안넘는지? 에서
→ 결혼을 했는지 안했는지

![2%20ML%20Pipel%2046a89/Untitled%2013.png](2%20ML%20Pipel%2046a89/Untitled%2013.png)

- MLFlow 같은 것

![2%20ML%20Pipel%2046a89/Untitled%2014.png](2%20ML%20Pipel%2046a89/Untitled%2014.png)

![2%20ML%20Pipel%2046a89/Untitled%2015.png](2%20ML%20Pipel%2046a89/Untitled%2015.png)

![2%20ML%20Pipel%2046a89/Untitled%2016.png](2%20ML%20Pipel%2046a89/Untitled%2016.png)

- pytorch는 serving할 때 별로 안좋음
→ 코드를 같이 서빙해야함
- gRPC 
→ 속도는 빠르나, 컴파일 프로세스가 어려움
    
    → 아직 성숙하지 않은 느낌
    

![2%20ML%20Pipel%2046a89/Untitled%2017.png](2%20ML%20Pipel%2046a89/Untitled%2017.png)

![2%20ML%20Pipel%2046a89/Untitled%2018.png](2%20ML%20Pipel%2046a89/Untitled%2018.png)

- 배포한 모델(Ground Truth : 어느한 장소에서 수집된 정보를 의미)의 input / output 다 받을 수 있어야 함

![2%20ML%20Pipel%2046a89/Untitled%2019.png](2%20ML%20Pipel%2046a89/Untitled%2019.png)