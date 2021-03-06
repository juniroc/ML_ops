# 6. Data Validation - Tensorflow Data Validation

# TFX

- 다양한 라이브러리 존재.
→ 상황에 따라 필요한 라이브러리를 선별적으로 이용

## TFDV

- Framework Independent 하기에 Pytorch 에서도 가능

![6%20Data%20Val%207c170/Untitled.png](6%20Data%20Val%207c170/Untitled.png)

### 데이터 검증이 필요한 이유?

![6%20Data%20Val%207c170/Untitled%201.png](6%20Data%20Val%207c170/Untitled%201.png)

- 심각한 경우는 1년 뒤에 뭔가 잘못되었다는 걸 인지
→ 모니터링을 하지 않는 경우
    
    → 즉,데이터가 정상적인지 확인하는 것이 중요
    

![6%20Data%20Val%207c170/Untitled%202.png](6%20Data%20Val%207c170/Untitled%202.png)

### Example

- 택시 팁 여부 데이터

![6%20Data%20Val%207c170/Untitled%203.png](6%20Data%20Val%207c170/Untitled%203.png)

![6%20Data%20Val%207c170/Untitled%204.png](6%20Data%20Val%207c170/Untitled%204.png)

![6%20Data%20Val%207c170/Untitled%205.png](6%20Data%20Val%207c170/Untitled%205.png)

![6%20Data%20Val%207c170/Untitled%206.png](6%20Data%20Val%207c170/Untitled%206.png)

![6%20Data%20Val%207c170/Untitled%207.png](6%20Data%20Val%207c170/Untitled%207.png)

![6%20Data%20Val%207c170/Untitled%208.png](6%20Data%20Val%207c170/Untitled%208.png)

![6%20Data%20Val%207c170/Untitled%209.png](6%20Data%20Val%207c170/Untitled%209.png)

- serving_data : 정답이 없는 실제 서빙할 데이터

![6%20Data%20Val%207c170/Untitled%2010.png](6%20Data%20Val%207c170/Untitled%2010.png)

![6%20Data%20Val%207c170/Untitled%2011.png](6%20Data%20Val%207c170/Untitled%2011.png)

![6%20Data%20Val%207c170/Untitled%2012.png](6%20Data%20Val%207c170/Untitled%2012.png)

![6%20Data%20Val%207c170/Untitled%2013.png](6%20Data%20Val%207c170/Untitled%2013.png)

- csv파일의 통계적 특성 파악

![6%20Data%20Val%207c170/Untitled%2014.png](6%20Data%20Val%207c170/Untitled%2014.png)

![6%20Data%20Val%207c170/Untitled%2015.png](6%20Data%20Val%207c170/Untitled%2015.png)

![6%20Data%20Val%207c170/Untitled%2016.png](6%20Data%20Val%207c170/Untitled%2016.png)

![6%20Data%20Val%207c170/Untitled%2017.png](6%20Data%20Val%207c170/Untitled%2017.png)

- 학습 데이터에 대한 스키마 추출 가능
→ 어떤 feature가 어떤 자료형인지(int, str, object...)
→ ex) 카테고리 : 남자, 여자

![6%20Data%20Val%207c170/Untitled%2018.png](6%20Data%20Val%207c170/Untitled%2018.png)

- 어느 데이터에 몰려있는지도 파악이 가능
→ 특이 데이터가 들어온 경우 anomaly 데이터로 인식

![6%20Data%20Val%207c170/Untitled%2019.png](6%20Data%20Val%207c170/Untitled%2019.png)

- 파란색 : 평가 데이터
- 노란색 : 학습 데이터

![6%20Data%20Val%207c170/Untitled%2020.png](6%20Data%20Val%207c170/Untitled%2020.png)

- Company 에 학습 데이터에는 없지만, 평가 데이터에만 있는 경우 anomaly..
- Payment_type 에서도 학습 데이터엔 없고, 평가 데이터에만 있는 Prcard 이 존재

→ 잘 섞이지 않은 것.

즉, 새로운 데이터가 존재 Anomaly 데이터 탐지.

![6%20Data%20Val%207c170/Untitled%2021.png](6%20Data%20Val%207c170/Untitled%2021.png)

![6%20Data%20Val%207c170/Untitled%2022.png](6%20Data%20Val%207c170/Untitled%2022.png)

- company에 대한 제한치를 낮춤
- payment_type 에 Prcard 를 추가해줌

![6%20Data%20Val%207c170/Untitled%2023.png](6%20Data%20Val%207c170/Untitled%2023.png)

![6%20Data%20Val%207c170/Untitled%2024.png](6%20Data%20Val%207c170/Untitled%2024.png)

- serving 데이터의 anomaly 파악

![6%20Data%20Val%207c170/Untitled%2025.png](6%20Data%20Val%207c170/Untitled%2025.png)

![6%20Data%20Val%207c170/Untitled%2026.png](6%20Data%20Val%207c170/Untitled%2026.png)

### 데이터 드리프트 및 스큐

- 모델의 성능이 하락하는 것을 어떻게 인지?
→ 들어온 input 값들의 분포가 바뀔 경우...?
ex) 남성 데이터가 95% 였으나 최근 여성에게 인기가 폭발하여 여성데이터 유입이 많아졌다
    
    → 데이터 드리프트 발생
    
    ※ 데이터 드리프트 : 입력데이터가 급격히 바뀌거나 할 경우 (표류)
    

![6%20Data%20Val%207c170/Untitled%2027.png](6%20Data%20Val%207c170/Untitled%2027.png)

![6%20Data%20Val%207c170/Untitled%2028.png](6%20Data%20Val%207c170/Untitled%2028.png)

- 연속형, 카테고리 형 데이터 모두 체크 가능

![6%20Data%20Val%207c170/Untitled%2029.png](6%20Data%20Val%207c170/Untitled%2029.png)

※ **스큐** : 데이터 편향

![6%20Data%20Val%207c170/Untitled%2030.png](6%20Data%20Val%207c170/Untitled%2030.png)

![6%20Data%20Val%207c170/Untitled%2031.png](6%20Data%20Val%207c170/Untitled%2031.png)

![6%20Data%20Val%207c170/Untitled%2032.png](6%20Data%20Val%207c170/Untitled%2032.png)

- **분포 스큐** : 일반적으로 말한 것, 30대가 많다가 갑자기 40~50대가 급증한 경우

![6%20Data%20Val%207c170/Untitled%2033.png](6%20Data%20Val%207c170/Untitled%2033.png)

---

## 실습링크

[Google Colaboratory](https://link.chris-chris.ai/ai-lecture-12)