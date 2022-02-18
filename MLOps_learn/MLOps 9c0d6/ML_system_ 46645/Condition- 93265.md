# Condition-based serving pattern

- 상황에 따라 예측 할 대상이 달라지는 경우
- 룰 베이스 방식으로 상황에 따라 모델 선택 할 경우

---

- 상황에 따라 모델을 선택하는 구조
ex) 사용자의 상태
- 시간 (7am~11am, 11am~7pm, 7pm~7am)
- 장소 (Asia, Europe, America)
- 상황 등
- 각 모델에 대한 선택과 분산 처리는 프록시를 통해 제어

![Condition-%2093265/Untitled.png](Condition-%2093265/Untitled.png)

**장점**

- 상황에 따라 알맞은 모델 제공

**단점**

- 모델 수에 따라 운영 비용 증가