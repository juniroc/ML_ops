# Data model versioning pattern

- 데이터와 모델 버전 관리해야 하는 경우
(실제 운영 환경에서 머신러닝을 사용해야 하는 대부분의 경우)

---

### 모델 **구성요소**

- 머신러닝 사용 목적
- 가치 (주가 예측, 이상징후 감지, 얼굴 인식 등)
- 기능 (분류, 회귀, 랭킹 등)
- 릴리즈  버전
- 사전 릴리즈
- 릴리즈
- 입/출력 데이터 및 인터페이스
- 입/출력 데이터 타입과 구조
- 코드와 파이프라인
- 학습 코드와 파이프라인
- 예측 코드와 파이프라인
- 코드와 파이프라인을 실행할 환경 : 라이브러리, OS, 언어, 인프라, 버젼 등
- 하이퍼파라미터
- 알고리즘이나 라이브러리에 의존
- 데이터
- 학습, 검증, 테스트 셋 분리

---

위 구성 요소들의 버전을 관리해야 함.

1. 모델 이름
- **`Mdoel-name**_x.y.z-data.split` 
- '가치'나 '기능'에 따라 정의하면 보기 좋음

2. 릴리즈 버전 
- `model-name_**X**.y.z-data.split` 
- 사전릴리즈 모델 : 0
- 릴리즈 후 모델 : 1

3. 인터페이스 버전 : 입출력 인터페이스 버전
- `model-name_x.**Y**.z-data.split`
- 입출력 구조가 변경되면 외부 인터페이스도 함께 변경
- 인터페이스, 데이터 타입/구조는 Git 같은 코드 저장소 이용해 관리
- 0 부터 시작

4. 로직 버전 : 알고리즘, 라이브러리, 하이퍼파라미터 등
- `model-name_x.y.**Z**-data.split`
- 인터페이스 변경 없는 전처리, 알고리즘, 라이브러리, 파라미터의 버저닝 담당
- 변경사항은 코드 레파지토리에서 브랜치로 관리

5. 데이터 검색 버전 : 데이터 검색 방법 / 데이터 저장 버전 : 데이터 분할 및 보완할 DWH 버전
- 검색 버전 : `model-name_x.y.z-**DATA**.split` 
- 일반적으로 데이터 DWH에 저장
- 학습용, 테스트용, 평가용 데이터 따로 나누어 저장
- 저장 버전 : `model-name_x.y.z-data.**SPLIT`** 

![Data%20model%2047894/Untitled.png](Data%20model%2047894/Untitled.png)

**장점**

- 모델 버전 관리 가능
- 재현을 위한 각종 요건 관리
- 재학습 및 업데이트 관리

**단점**

- 코드, 모델, 데이터들을 여러 환경에서 관리해야 함