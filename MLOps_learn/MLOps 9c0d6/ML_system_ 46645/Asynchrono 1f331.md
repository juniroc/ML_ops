# Asynchronous Pattern

- 프로세스와 예측 사이 의존성이 없는 경우
- 예측 요청할 클라이언트와 응답할 목적지가 분리되어 있는 경우

---

- 특정한 경우 `(diagram2)`클라이언트가 예측 지연을 기다리지 않아도 됨.

![Asynchrono%201f331/Untitled.png](Asynchrono%201f331/Untitled.png)

![Asynchrono%201f331/Untitled%201.png](Asynchrono%201f331/Untitled%201.png)

**장점**

- 클라이언트와 예측 분리
- 예측 대기 시간을 기다릴 필요 없음

**단점**

- 큐, 캐시 또는 유사한 종류의 프록시 필요
- 실시간 예측엔 부적절