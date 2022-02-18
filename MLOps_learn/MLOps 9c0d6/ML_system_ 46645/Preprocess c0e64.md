# Preprocess-prediction pattern

- 전처리 및 예측에서 사용하는 코드가 다른 경우

---

- 보통 전처리에서 normalization, standardization, one-hot encoding, image 크기 조정 등 실행
    
    → 해당 과정은 인퍼런스 할 경우에도 양식을 맞춰야하는 문제 발생
    
- 전처리와 예측 모델을 별도의 리소스로 나눔
→ 복잡, 각각 버전을 고려

![Preprocess%20c0e64/Untitled.png](Preprocess%20c0e64/Untitled.png)

- 아래는 전처리와 예측 앞에 프록시를 배치해 마이크로서비스화

![Preprocess%20c0e64/Untitled%201.png](Preprocess%20c0e64/Untitled%201.png)

**장점**

- 리소스 효율적 사용
- 구성 요소별 라이브러리 및 언어 버전 선택 가능
- 유연성 및 확장성

**단점**

- 전처리와 예측 사이에 병목 현상 일어날 수 있음