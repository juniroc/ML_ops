# Review_

### Pipeline의 필요성

- **why?**
    - 배포하고 **관리**하지 **않으면** **성능이 떨어짐**
    → 이를 시스템화하여 보완 가능 (유지보수)
    - 모델링은 전체의 **5%**정도
    → 사실상 모델 성능 개선 및 유지보수에서 많은 시간이 소요됨
    → 이를 수월하게 도움
    - 특히 어려운 문제
    - 데이터가 빠르게 변화 (주 단위) (TFDV)
       → 모델 **성능 저하 빈번**하게 발생
    - ML == **실험**
    - 여러 Feature, 알고리즘 및 파라미터 구성을 시도해 **가장 적합한 것을 찾아야** 함
    - 즉 무엇이 효과가 있었는지, 추적하고 코드 **재사용을 극대화 및 재현성**이 중요
    - 명확하지 않은 배포 방식
    → 모델을 학습이 **잘 동작하는지 체크하는 방식이 명확하지 않음**
        
        → 트리거 포인트가 여러개
          ex) 성능을 어느정도 판단한 이후, 성능이 어느정도 떨어지면 재학습 후, daily 단위로, 온라인 실시간 학습 이후 또는 데이터 분포의 중요한 변화시 (수학적 분포가 존재할 경우)
        

---

### Pipeline 의 단계

![Review_%20a04d9/Untitled.png](Review_%20a04d9/Untitled.png)

- **Pipeline 의 단계**
    
    1. **수집 / 버져닝**
    → 데이터 수집
    → 데이터가 변화하며 들어오기 때문에 **Versioning 중요**
    
    2. **유효성 검사**
    → 새 데이터의 통계가 예상대로 들어왔는지 확인
    ex)범주 범위 또는 분포를 통해 이상 징후가 감지될 경우 경고해줄 수 있음.
    → TFDV 로 쉽게 파악 가능
    
    3. **전처리**
    → 흔히 알고 있는 전처리
    ex) feature engineering, one-hot, normalization 등
    
    4. **학습/튜닝**
    → 이 과정에서 하이퍼파라미터 들어간다 보면됨
    → Pipeline 의 꽃 == AutoML
    → 이때 Automl을 아무리 돌려도 성능이 고정되는 경우 존재, **early stopping** 고려
    
    5. **모델 분석**
    → 기존 MSE, AUC, Recall 등
    → 넘어서, TFX 의 TFMA 이용하면 각 feature별 결과에 어떤 영향을 미치는지 파악 가능
    
    6. **모델 버젼관리**
    → MLFlow, BentoML 등 이용해 모델의 버젼 업데이트 생각
    
    7. **모델 배포
    →** 앞의 단계를 거쳐 괜찮은 모델 배포
    
    8. **피드백 (루프 반복)
    →** 전체 파이프라인 자동화 한 것 유지보수
    

---

### W&B (Weights and Biases)

- **Explain**
    - 본인이 느끼기에 nni와 상당한 부분이 겹치는 서비스
    
    - 실험관리가 필요한 이유
    → 지난 학습시 **최적 파라미터**?
    → 썻던 논문, **실험결과 재연이 안되는 경우**..
    → **어떤 데이터로 학습** 시킨건지?
    → 가장 **성능 좋은 모델**은 무엇?, 어떤 **Metric** (**Acc? AUC?**)
    → **어떤 하이퍼파라미터가 가장 영향**이 큰지?
    
    ![Review_%20a04d9/Untitled%201.png](Review_%20a04d9/Untitled%201.png)
    
    **쉬운 사용법**
    
    ![Review_%20a04d9/Untitled%202.png](Review_%20a04d9/Untitled%202.png)
    
    ![Review_%20a04d9/Untitled%203.png](Review_%20a04d9/Untitled%203.png)
    
    **학습 시각화**
    
    ![Review_%20a04d9/Untitled%204.png](Review_%20a04d9/Untitled%204.png)
    
    **GPU 사용률 모니터링**
    
    ![Review_%20a04d9/Untitled%205.png](Review_%20a04d9/Untitled%205.png)
    
    - 사실상 **NNI 와 비슷**하나 차이점은 **GPU 사용률 모니터링**이 가능하다는 점
    - 가장 큰 문제는 기업 단위로 쓸 경우 비용이 많이 듬 (**인당 20만원)
    → nni == Open_source**
    

---

### 리서치 코드 품질 관리 (feat. 협업)

- **Problem**
    1. 사실상 리서치 코드는 각자 개인 컴퓨터에 저장
    2. '**Ctrl-c + Ctrl-v**'(복붙) 기반
    3. paper 위주 코드는 재연이 불가능한 경우 많음
    4. 나만 알아볼 수 있는 변수명

- **1) Black**
    
    
    `requirements.txt` file  ****
    
    ```python
    black==19.10b0
    coverage==4.4.1
    codeclimate-test-reporter==0.2.3
    
    ### pip install -r requirements.txt 이용해서 다운로드
    ```
    
    `[main.py](http://main.py)` file
    
    ```python
    ### main.py 파일
    
    def      helloworld(a):
        print(f"hello world! {a}!")#hmm..
    
    if __name__ ==    "__main__":
        helloworld("nujnim")
    ```
    
    - 위 `[main.py](http://main.py)` 파일은 임의로 지저분하게 만든 것
    
    ```bash
    black main.py
    ```
    
    - 위 명령어를 실행하면
    
    `main.py`
    
    ```python
    def helloworld(a):
        print(f"hello world! {a}!") #hmm..
    
    if __name__ == "__main__":
        helloworld("nujnim")
    ```
    
    - 위와 같이 깔끔하게 변함

- **2) Lint**
    
    `lint.yml`
    
    ```yaml
    ### .github/workflows/lint.yml 파일
    
    name: Lint Code Base
    
    ## push가 일어났을 때
    on: push
    
    jobs:
      # Set the job key. The key is displayed as the job name
      # when a job name is not provided
      super-lint:
        # Name the Job
        name: Lint Code Base
        # Set the type of machine to run on
        runs-on: ubuntu-latest
    
        env:
          OS: ${{ matrix.os }}
          PYTHON: '3.7'
    
        steps:
          # Checks out a copy of your repository on the ubuntu-latest machine
          - name: Checkout code
            uses: actions/checkout@v2 # 코드 체크아웃
    
          # Runs the Super-Linter action
          - name: Lint Code Base
            uses: github/super-linter@v3
            env:
              DEFAULT_BRANCH: master
              GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
              VALIDATE_PYTHON_BLACK: true
              VALIDATE_PYTHON_FLAKE8: true
    ```
    
    - push가 일어났을 때
    → 우분투 파이썬 3.7 에서 코드 체크 아웃, black과 flake8 실행 (True 설정)
        
        → **flake8** : python 타입 체크 라이브러리
        
    
    ```bash
    git add.
    
    git status
    
    ### git config --global user.email "lmj35021@gmail.com"
    ### git config --global user.name "juniroc"
    
    ### 위는 따로 설정
    git commit -a -m "main.py implemented"
    
    # 브런치 생성 후 이동
    git checkout -b feature/210521-main
    
    # 푸쉬
    git push --set-upstream origin feature/210521-main
    ```
    
    - 위 코드를 통해 코드를 업데이트
    
    ![5%20Managing%20b5ec5/Untitled%2029.png](5%20Managing%20b5ec5/Untitled%2029.png)
    
    ![5%20Managing%20b5ec5/Untitled%2030.png](5%20Managing%20b5ec5/Untitled%2030.png)
    
    - 위와 같이 CI 진행 (자동으로 이미지 pulling)
    
    ![5%20Managing%20b5ec5/Untitled%2031.png](5%20Managing%20b5ec5/Untitled%2031.png)
    
    - 앗 에러 발생
    
    ![5%20Managing%20b5ec5/Untitled%2032.png](5%20Managing%20b5ec5/Untitled%2032.png)
    
    - **73~74**가 **75~80 코드**로 변환되어야 함을 알림
    
    ![5%20Managing%20b5ec5/Untitled%2033.png](5%20Managing%20b5ec5/Untitled%2033.png)
    
    - Push Request를 막음
    
    **요약** : **협업을 위한 툴**이며, 큰 문제로는 **유료**.
    

---

### TFDV(Tensorflow Data Validation)

- **데이터 검증 라이브러리**
    - Train_dataset validation
    
    ![Review_%20a04d9/Untitled%206.png](Review_%20a04d9/Untitled%206.png)
    
    ![Review_%20a04d9/Untitled%207.png](Review_%20a04d9/Untitled%207.png)
    
    - Evaluation dataset Validation
    
    ![Review_%20a04d9/Untitled%208.png](Review_%20a04d9/Untitled%208.png)
    
    ![Review_%20a04d9/Untitled%209.png](Review_%20a04d9/Untitled%209.png)
    
    - Train / Eval dataset 확인
    - Company column내  Train dataset 에는 없고, Eval dataset에 존재 
    - 즉 ,새로운 데이터가 존재 ( Anomaly 데이터 탐지)
        
        → 데이터 셋이 잘 섞이지 않은 것.
        
    
    ![Review_%20a04d9/Untitled%2010.png](Review_%20a04d9/Untitled%2010.png)
    
- **데이터 드리프트 및 스큐**
    - 모델의 성능이 하락하는 것을 어떻게 인지?
    → 들어온 input 값들의 분포가 바뀔 경우...?
    ex) 남성 데이터가 95% 였으나 최근 여성에게 인기가 폭발하여 여성데이터 유입이 많아졌다
        
        → 데이터 드리프트 발생
        
        ※ **데이터 드리프트** : 입력데이터가 급격히 바뀌거나 할 경우 (표류)
        
        ※ **데이터 스큐 :** 데이터 편향
        
    
    - 데이터가 급격히 변하는 컬럼의 value 를 anomaly 로 뽑음
    → train_data 와 serving_data  비교
    
    ![Review_%20a04d9/Untitled%2011.png](Review_%20a04d9/Untitled%2011.png)
    
    - L-Infinity Distance 로 두 데이터의 분포차이를 계산
    → Threshold 보다 크게 차이나는 column을 찾아냄
    + 가장 크게 차이나는 value 도 찾아냄

---

### WIT (What-If-Tool)

- **데이터 EDA 시각화**
    - 학습된 모델에서 가장 **해당 피쳐에 대한 상관관계 등** **시각적**으로 확인할 수 있다.
    
    ![Review_%20a04d9/Untitled%2012.png](Review_%20a04d9/Untitled%2012.png)
    
    - **TFDV**에서 이용했던 **Feature에 대한 분포 및 특징 단순화**
    
    ![Review_%20a04d9/Untitled%2013.png](Review_%20a04d9/Untitled%2013.png)
    
    ![Review_%20a04d9/Untitled%2014.png](Review_%20a04d9/Untitled%2014.png)
    
    [Jupyter Notebook](http://223.194.90.113:8080/notebooks/ML_Ops/Inflearn/What_if_Tool/210531_what_if_tool_example.ipynb#Invoke-What-If-Tool-for-test-data-and-the-trained-model)
    

---

### Kubeflow

- **Kubernetes 기반** **ML 시스템 구성 요소 배열**하기 위한 **플랫폼**
- **필요한 이유**
    
    ![Review_%20a04d9/Untitled%2015.png](Review_%20a04d9/Untitled%2015.png)
    

- **Kubeflow Pipeline**
    
    ![Review_%20a04d9/Untitled%2016.png](Review_%20a04d9/Untitled%2016.png)
    
    - python_func 함수에다 decorator 를 달아주는 방식을 컴포넌트 생성
    
    ![Review_%20a04d9/Untitled%2017.png](Review_%20a04d9/Untitled%2017.png)
    
    - 마지막으로 **@kfp.dsl.pipeline** 데코레이터 활용해 kubeflow 컴포넌트를 파이프라인으로 패키징
    
    ![Review_%20a04d9/Untitled%2018.png](Review_%20a04d9/Untitled%2018.png)
    
    - 여러 **컴포넌트를 조합**하여 **파이프라인 생성**
        
        → **파이프라인에 빌드**하면 **Yaml 파일 생성**
        → 이를 다시  Zip, Tar로 패키징 가능
        → webui 파이프라인 등록할 수 도 있음
        
        → CLI 로 등록도 가능
        

---

###