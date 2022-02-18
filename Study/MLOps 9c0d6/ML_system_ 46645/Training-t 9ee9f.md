# Training-to-serving pattern

- 모델~운영 전체 워크플로우 디자인하는 경우
- 학습 직후 모델 출시할 경우
- 모델~운영 환경 자동으로 배포할 경우
- 안정적으로 학습할 수 있을 경우
- 자주 업데이트해야 할 경우

---

- 학습 패턴과 서빙 패턴을 조합하는 구성
- 학습 파이프라인이 완료되면 자동으로 릴리즈하여 모델 서버를 구축할 수 있음
- 학습 파이프라인
- `batch training pattern` or `pipeline training pattern` 선택
- 서빙 패턴
- `model load pattern` : 현재 서버에서 변경 없이 모델 업데이트
- `model-in-image pattern` : 모든 서버를 업데이트
- `microservice horizontal pattern` : 새로운 예측 서버를 다른 서버와 병렬 배치, 프록시를 통해 서비스 검색하여 예측 서버들과 연결
- **서비스 관리 관점**
- `prediction log pattern` and `prediction monitoring pattern` 사용 필수
- **학습 후 자동**으로 **모델 릴리즈**하고 실제 서비스에 투입
- 반드시 **학습과 평가가 안정적**이고 **학습 파이프라인이 안정적으로 가동**
- 실제 운영 환경에 있는 모든 모델을 항상 가동해 둘 필요가 있는지 검토
→ 모델이 (오래되거나, 성능 저하로) 필요하지 않는 경우 서비스에서 제외

![Training-t%209ee9f/Untitled.png](Training-t%209ee9f/Untitled.png)

**장점**

- 학습 후 바로 서비스에 릴리즈 가능
- 릴리즈를 자주 가능

**단점**

- 파이프라인, 자동 릴리즈, 서비스 디스커버리 등을 개발해야 함
- 학습 결과가 불안정할 경우 부적합