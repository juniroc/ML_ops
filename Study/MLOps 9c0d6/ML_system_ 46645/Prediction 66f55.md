# Prediction log pattern

- 개선을 위한 예측, 지연 시간 등 로그를 사용할 경우
- 방대한 로그량에 DWH 부하가 우려될 경우
- 로그를 사용해 모니터링 및 알림을 보내고 싶을 때

---

- 입력, 지연 시간, 이벤트, 예측 결과 및 관련 활동 등 로그로 수집
- 하나의 DWH 에 통합 관리하는 것이 효율적
- 큐 방식 사용 시 DWH 부하를 줄일 수 있음
- 예측이나 클라이언트 로그가 과거 패턴과 확연히 달라지는 경우 - 이상이 생김을 예측
→ 알림
- **예측 시스템 로그**뿐만이 아닌 **클라이언트 로그**에 대해서도 **이상 상태 정의/알림 대상** 설정

![Prediction%2066f55/Untitled.png](Prediction%2066f55/Untitled.png)

**장점**

- 예측 효과 분석 가능
- 경고 알림

**단점**

- 로그 양에 따라 비용 증가 가능성