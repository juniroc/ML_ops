# 1. ML Pipeline

![1%20ML%20Pipel%20381c2/Untitled.png](1%20ML%20Pipel%20381c2/Untitled.png)

![1%20ML%20Pipel%20381c2/Untitled%201.png](1%20ML%20Pipel%20381c2/Untitled%201.png)

- 배포하고 관리하지 않으면 성능이 현저하게 떨어짐.
→ 이를 보완 가능하게 함
→ 시스템화 하여 컨트롤이 가능하도록 함

![1%20ML%20Pipel%20381c2/Untitled%202.png](1%20ML%20Pipel%20381c2/Untitled%202.png)

- ML 시스템 개발 및 배포는 비교적 쉽고 빠름
→ 해당 시스템을 유지하고 관리하는 비용은 매우 큼
- 그리고 지금 당장은 그냥 배포하는게 편하지만 나중을 본다면 **결국 큰 부채**
→ 기술 부채

![1%20ML%20Pipel%20381c2/Untitled%203.png](1%20ML%20Pipel%20381c2/Untitled%203.png)

![1%20ML%20Pipel%20381c2/Untitled%204.png](1%20ML%20Pipel%20381c2/Untitled%204.png)

[Refactoring이란?](https://nesoy.github.io/articles/2018-05/Refactoring)

### ML pipeline의 필요성

![1%20ML%20Pipel%20381c2/Untitled%205.png](1%20ML%20Pipel%20381c2/Untitled%205.png)

![1%20ML%20Pipel%20381c2/Untitled%206.png](1%20ML%20Pipel%20381c2/Untitled%206.png)

![1%20ML%20Pipel%20381c2/Untitled%207.png](1%20ML%20Pipel%20381c2/Untitled%207.png)

- 일, 주 단위 데이터가 주어질 경우
- 오프라인 서빙 → 동작실패 경우 즉시 확인 가능

### 머신러닝 프로그래밍

![1%20ML%20Pipel%20381c2/Untitled%208.png](1%20ML%20Pipel%20381c2/Untitled%208.png)

### 머신러닝 프로그래밍 문제의 특징

![1%20ML%20Pipel%20381c2/Untitled%209.png](1%20ML%20Pipel%20381c2/Untitled%209.png)

- 고정된 데이터셋
    
    → ex) MNIST는 이미지 사이즈가 다 같지만, 실제 데이터는 이미지 사이즈가 다른 경우가 대부분
    
- 검증되지 않은 데이터셋
→ ex) '나이' 컬럼에 -값(음수) 데이터 인 경우
- 모델이 잘못된 경우
→ 알기가 어려움 → output 결과를 뽑아봐야함(inference)

![1%20ML%20Pipel%20381c2/Untitled%2010.png](1%20ML%20Pipel%20381c2/Untitled%2010.png)

![1%20ML%20Pipel%20381c2/Untitled%2011.png](1%20ML%20Pipel%20381c2/Untitled%2011.png)

![1%20ML%20Pipel%20381c2/Untitled%2012.png](1%20ML%20Pipel%20381c2/Untitled%2012.png)

- 예측값이 정확한지 아닌지 알 방법이 없는 경우가 많음. (ex, 주식)

![1%20ML%20Pipel%20381c2/Untitled%2013.png](1%20ML%20Pipel%20381c2/Untitled%2013.png)

- 모델뿐만이 아닌, 전처리, 후처리, 관련 아키텍트도 있음, 배포하는 CI/CD이 다름.
→ software는 merge해서 배포하지만,
→ ML에서는 모델 학습이 잘 동작하는지 체크하는 방식이 명확하지 않음.
→ 즉, 트리거 포인트가 여러개..
ex) 성능을 어느정도 판단한 이후에.. , 성능이 어느정도 떨어지면 재학습 후, daily 단위로, 온라인 실시간 학습 이후

![1%20ML%20Pipel%20381c2/Untitled%2014.png](1%20ML%20Pipel%20381c2/Untitled%2014.png)

![1%20ML%20Pipel%20381c2/Untitled%2015.png](1%20ML%20Pipel%20381c2/Untitled%2015.png)

- 데이터 분포가 변화하였다는 수학적 분포가 존재

### 진화하는 데이터셋과 Metric

![1%20ML%20Pipel%20381c2/Untitled%2016.png](1%20ML%20Pipel%20381c2/Untitled%2016.png)

- 지속적 학습이 되는 곳이 많지는 않음.

![1%20ML%20Pipel%20381c2/Untitled%2017.png](1%20ML%20Pipel%20381c2/Untitled%2017.png)

![1%20ML%20Pipel%20381c2/Untitled%2018.png](1%20ML%20Pipel%20381c2/Untitled%2018.png)

![1%20ML%20Pipel%20381c2/Untitled%2019.png](1%20ML%20Pipel%20381c2/Untitled%2019.png)

![1%20ML%20Pipel%20381c2/Untitled%2020.png](1%20ML%20Pipel%20381c2/Untitled%2020.png)

![1%20ML%20Pipel%20381c2/Untitled%2021.png](1%20ML%20Pipel%20381c2/Untitled%2021.png)

![1%20ML%20Pipel%20381c2/Untitled%2022.png](1%20ML%20Pipel%20381c2/Untitled%2022.png)

![1%20ML%20Pipel%20381c2/Untitled%2023.png](1%20ML%20Pipel%20381c2/Untitled%2023.png)