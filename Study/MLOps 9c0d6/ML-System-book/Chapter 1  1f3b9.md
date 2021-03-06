# Chapter 1. ML system pattern화

## ML system pattern화

### 학습

- ‘어떻게 학습할 것인가’ 만큼 ‘언제 학습할 것인가' 역시 중요함
→ 새로운 데이터를 얻을 수 있다고 해서 매일 모델을 학습하는 것은 적절하지 않음
- 모델의 성능을 측정하고 평가한 뒤 정함.
→ 보통 시간이 지남에 따라 데이터 양상도 바뀌는 경우가 있어 성능이 떨어짐
→ 이때, 재학습이 필요
- 즉, 데이터의 패턴이 바뀌는 타이밍을 잘 파악해야하고, 파악이 어려운 모델의 경우 모델의 평가 기준을 세우고 그 평가에 따라 유연하게 학습

### 릴리즈

- 크게 1. `서버 사이드` , 2. `에지 사이드` 로 나뉨
- 서버 사이드 : 클라우드나 데이터 센터 백엔드 시스템에 추론기 배치
→ 에지 사이드에 비해 풍부한 연산 자원 사용 가능 (대부분이 여기에 해당)
- 에지 사이드 : 스마트폰이나 디바이스, 브라우저 같이 사용자의 수중에 있는 단말기에 추론기 설치 후 추론
→ 전력은 서버 사이드보다 작지만, 네트워크 매개하지 않아 고속 추론 가능

### 품질관리

- 실제로 모델이 가동되고 있을 경우 A/B 테스트를 통해 모델 및 추론기의 품질을 검증하는 것이 좋음
→ 이때 결과의 상태뿐만 아니라, **지연시간** 또는 추론에 대한 **사용자의 행동**까지도 고려해야 함
- 릴리즈 이후에도 모델을 지속해서 평가해야 함

---