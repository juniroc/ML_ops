# Pipeline training Pattern

- Batch Training Pattern 의 응용 패턴
- 각 작업을 개별 리소스로 분할 (server, container, worker)
→ 개별적으로 구축해 작업 실행과 재시도를 유연하게 실행
- 모든 작업은 개별 리소스로 배치
→ 의존 관계가 있는 작업 후에 실행 
즉, 이전 작업 실행 결과는 후속 작업에 제공(후속 작업의 인풋 데이터)
- Fault Tolerance(시스템 일부에서 결함 발생하여도 정상적/부분적으로 기능 수행) 위해 완료 데이터는 DWH 에 저장할 수도 있음
- 이전 작업 완료 즉시 다음 작업을 진행할 필요 없음
- 작업 워크플로우와 리소스 관리가 복잡함
- 각 작업의 독립성을 높임
→ 실행 조건이나 리소스 선택 등을 개별적 고려

![Pipeline%20t%2011483/Untitled.png](Pipeline%20t%2011483/Untitled.png)

**장점**

- 리소스나 라이브러리를 유연하게 선택
- 장애 분리
- 독립성을 높임
- 워크플로우 또는 **데이터 기반**의 작업 관리

**단점**

- 다중 구조로 여러 작업들을 관리해야 함
- 시스템 및 워크플로우가 복잡