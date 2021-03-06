# Web-single-pattern

- 가장 간단한 아키텍처
- 예측 서버를 빠르게 출시

---

- 예측 모델을 위한 모든 artifact 를 웹 서버에 함께 저장
- 단일 서버 REST(or GRPC) 인터페이스, 전처리, 훈련된 모델을 한 곳에서 사용
→ 간단히 생성 및 배포 가능
- 여러 복제본 배포할 경우
→ Load Balancer / Proxy 사용해 배포
- 인터페이스에 GRPC 사용하는 경우, 클라이언트측 로드 밸런싱 or L7 로드 밸런서 고려

---

웹 서버에 모델을 빌드하려면

---

![Web-single%208cd7b/Untitled.png](Web-single%208cd7b/Untitled.png)

**장점**

- 웹 서버, 전처리, 예측할 때 파이썬 같은 하나의 프로그래밍 언어 사용
- 아키텍처의 단순함
→ 관리 쉬움
- 트러블슈팅이 쉬움
- 동기식 시스템에서 모델 배포할 경우 웹 단일 패턴으로 시작하는 것이 좋음

**단점**

- 모든 구성요소가 서버 또는 도커 이미지에 저장
→ 패치하려면 전체 업데이트가 필요.