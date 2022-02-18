# 7. WIT : What-If-Tool

TFX - TFMA 라는 툴이 존재하긴 함 - model analysis

→ WIT 이 모델 분석의 상위 호환이라 볼 수 있음

![7%20WIT%20What%20c1069/Untitled.png](7%20WIT%20What%20c1069/Untitled.png)

![7%20WIT%20What%20c1069/Untitled%201.png](7%20WIT%20What%20c1069/Untitled%201.png)

머신러닝 SW 개발은 - 디버깅이 어려움

![7%20WIT%20What%20c1069/Untitled%202.png](7%20WIT%20What%20c1069/Untitled%202.png)

![7%20WIT%20What%20c1069/Untitled%203.png](7%20WIT%20What%20c1069/Untitled%203.png)

- 3가지 탭 존재

![7%20WIT%20What%20c1069/Untitled%204.png](7%20WIT%20What%20c1069/Untitled%204.png)

![7%20WIT%20What%20c1069/Untitled%205.png](7%20WIT%20What%20c1069/Untitled%205.png)

![7%20WIT%20What%20c1069/Untitled%206.png](7%20WIT%20What%20c1069/Untitled%206.png)

- 탭별 기능 존재

![7%20WIT%20What%20c1069/Untitled%207.png](7%20WIT%20What%20c1069/Untitled%207.png)

- Feature의 값을 바꾸었을 때 → 예측의 값도 바뀐다.
→ 예측의 변경도 확인 가능(실시간으로)

![7%20WIT%20What%20c1069/Untitled%208.png](7%20WIT%20What%20c1069/Untitled%208.png)

- Thresholding 을 카테고리별로 가능

![7%20WIT%20What%20c1069/Untitled%209.png](7%20WIT%20What%20c1069/Untitled%209.png)

---

## 실습

![7%20WIT%20What%20c1069/Untitled%2010.png](7%20WIT%20What%20c1069/Untitled%2010.png)

![7%20WIT%20What%20c1069/Untitled%2011.png](7%20WIT%20What%20c1069/Untitled%2011.png)

![7%20WIT%20What%20c1069/Untitled%2012.png](7%20WIT%20What%20c1069/Untitled%2012.png)

![7%20WIT%20What%20c1069/Untitled%2013.png](7%20WIT%20What%20c1069/Untitled%2013.png)

![7%20WIT%20What%20c1069/Untitled%2014.png](7%20WIT%20What%20c1069/Untitled%2014.png)

![7%20WIT%20What%20c1069/Untitled%2015.png](7%20WIT%20What%20c1069/Untitled%2015.png)

![7%20WIT%20What%20c1069/Untitled%2016.png](7%20WIT%20What%20c1069/Untitled%2016.png)

- 어떤 요소가 Classification에 영향을 미쳤는지 확인 할 수 있음
→ 이를 통해 misclassification 된 데이터를 디버깅할 수 있음.
    
    ex) age나, education을 바꿔보면서 수입이 증가하는지 확인 가능
    

- L1 distance → 절대값을 더한 것 (맨하탄)
- L2 distance → 제곱값을 더한 것 (유클리드)

![7%20WIT%20What%20c1069/Untitled%2017.png](7%20WIT%20What%20c1069/Untitled%2017.png)

![7%20WIT%20What%20c1069/Untitled%2018.png](7%20WIT%20What%20c1069/Untitled%2018.png)

- 왼쪽에 두개가 가장 가깝다고 찾아줌

![7%20WIT%20What%20c1069/Untitled%2019.png](7%20WIT%20What%20c1069/Untitled%2019.png)

- 어떤 데이터 포인트 변경이 모델 예측에 영향을 미치는지 탐구

![7%20WIT%20What%20c1069/Untitled%2020.png](7%20WIT%20What%20c1069/Untitled%2020.png)

![7%20WIT%20What%20c1069/Untitled%2021.png](7%20WIT%20What%20c1069/Untitled%2021.png)

![7%20WIT%20What%20c1069/Untitled%2022.png](7%20WIT%20What%20c1069/Untitled%2022.png)

![7%20WIT%20What%20c1069/Untitled%2023.png](7%20WIT%20What%20c1069/Untitled%2023.png)

- Thresholding을 실시간으로 확인하면서 정할 수 있음

![7%20WIT%20What%20c1069/Untitled%2024.png](7%20WIT%20What%20c1069/Untitled%2024.png)

- Cost Ratio (FP/FN) 을 정해 최적화 임계 값 최적화

![7%20WIT%20What%20c1069/Untitled%2025.png](7%20WIT%20What%20c1069/Untitled%2025.png)

---

## 실습 코드

[Jupyter Notebook](http://223.194.90.113:8080/notebooks/ML_Ops/Inflearn/What_if_Tool/210531_what_if_tool_example.ipynb#Invoke-What-If-Tool-for-test-data-and-the-trained-model)

- 자체 서버용

[Google Colaboratory](https://colab.research.google.com/drive/16Jbt3yzIHW-ED-pDSDFclnzD4fIkZhSR?usp=sharing#scrollTo=YyLr-_0de1Ii)

- 코랩

모델과 모델간의 비교도 가능