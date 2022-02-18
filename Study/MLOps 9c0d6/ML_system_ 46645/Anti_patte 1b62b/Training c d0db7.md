# Training code in serving pattern

- 학습, 실험, 평가에만 사용되어야 하는 코드가 서빙 코드에 들어간 경우
- **학습을 위한 리소스**들은 **서빙용과** 따로 **분리
ex) CPU, GPU, RAM, 네트워크 및 스토리지**
    
    → GPU - 학습용 ,  CPU - 서빙용
    
- **공통 표준화할 부분**과, **분리할 부분을 구분**

---

**- 표준화해야 하는 부분
→ OS, 언어, 라이브러리 버전
→ 전처리, 예측, 후처리 등 사용되는 코드와 로직**

**- 분리되어야하는 부분**

→ 학습에 이용되는 코드와 로직

→ 학습을 위한 라이브러리나 리소스들(학습용 GPU, 대용량 CPU, RAM, Network, DWH)  의존성

![Training%20c%20d0db7/Untitled.png](Training%20c%20d0db7/Untitled.png)

**장점**

- 배치 예측에서는 학습과 서빙을 동일한 환경에서 실행시킬 수 있음.

**단점**

- 학습용과 서빙용 코드와 설정들을 분리하는 비용 필요
→ 당장은 비용이 발생하나, 장기적 사용시 보안 측면으로는 분리하는 것이 좋음.