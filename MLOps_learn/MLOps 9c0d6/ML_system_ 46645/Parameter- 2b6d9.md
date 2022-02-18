# Parameter-based serving pattern

- 파라미터를 사용해 예측을 제어하고 싶은 경우
- 룰 베이스로 제어가 가능한 경우

---

- 룰 베이스의 장치
- 환경 변수 등을 이용하여 해당 요청 제어

![Parameter-%202b6d9/Untitled.png](Parameter-%202b6d9/Untitled.png)

**장점**

- 엣지 케이스에 대한 비정상적인 예측을 피할 수 있음

**단점**

- 모든 케이스를 커버할 수는 없음
- 룰이 늘어남에 따라 운영의 복잡도가 올라갈 수 있음