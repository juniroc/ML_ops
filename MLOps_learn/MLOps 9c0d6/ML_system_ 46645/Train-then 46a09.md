# Train-then-serve pattern

- 모델링~운영 전체 워크플로우 디자인하는 경우
- 학습 / 릴리즈를 각각 다른 워크플로우로 분리할 경우
- 모델 릴리즈 품질을 수동으로 평가할 때
- 실제 운영 환경에서 모델을 평가할 때

---

- 학습 패턴과 서빙 패턴을 조합하는 구성
→ **평가가 끝난 학습된 모델**을 **수동**으로 **릴리즈**하는 방법
- 사람에 의한 평가가 들어감
→ 자주 모델 릴리즈하려면 적합하지 않음
→ 모델과 시스템의 품질을 확실히 보장
- 학습과 서빙을 연결하기 위해 **`model load pattern`**이나 **`model-in-image pattern`**을 사용
    
    → `model load pattern` : 현재 **서버에 변경 없이 모델을 업데이트**할 경우
    
    → `model-in-image pattern` : **전체 서버를 업데이트**할 경우
    
    → 또는 `parameter-based serving pattern` : 프록시 환경 변수 수정으로 운영 중인 모델의 예측 동작 업데이트
    
- 서비스 관리 측면에서 `prediction log pattern`과 `prediction monitoring pattern`은 반드시 사용
- 학습 / 서빙 단계의 분리 장점
→ 릴리즈 전 모델 평가 가능

![Train-then%2046a09/Untitled.png](Train-then%2046a09/Untitled.png)

**장점**

- 릴리즈 전 수동으로 모델 평가
- 워크플로우와 학습 및 릴리즈 장애 분리

**단점**

- 자동화가 안됨