# Chapter 3-3. DVC + CML 연계를 통한 Model Metric Tracking

### **Github Actions 를 활용하여 예제 머신러닝 코드 실행하여 성능 지표 출력하기**

- New Repository
    - Repository name : github-actions-cml
- 실습 Github Repo 복사
    - [https://github.com/yunjjun/github-actions-project.git](https://github.com/yunjjun/github-actions-project.git) 에 접속하여 github-actions-cml 브랜치로 이동하여 우측 상단 Fork 클릭
    - 또는 해당 링크 복사 후 자신의 Github Repo 에 push
        
        ```bash
        # 실습 Repo 데이터 다운로드
        # 로컬에 새 폴더 생성
        git init
        git remote add origin <git 저장소>
        git pull origin main
        git checkout -b main
        git config --global user.email <이메일 주소>
        git config --global user.name <이름>
        # 실습 Repo 의 데이터를 새 폴더로 복사
        git add .
        git commit -m "first commit"
        git push origin main
        ```
        
- [Red Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) 데이터 소개
- 훈련 코드 (train.py) 소개
- .github/workflows/cml.yml 파일 생성
    - 아래 코드 복사
    
    ```yaml
    name: model-training
    on: [push]
    jobs:
      run:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v2
          - uses: actions/setup-python@v2
          - uses: iterative/setup-cml@v1
          - name: Train model
            env:
              REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            run: |
              pip install -r requirements.txt
              python train.py
    					echo "MODEL METRICS"
    					cat metrics.txt
    ```
    
- 새로운 브랜치 만들어서 commit →  build → terminal 에 metric 프린트 → 충분하지 않음

![Untitled](Chapter%203-%206aa3d/Untitled.png)

- 위 사진처럼 `.github/workflows` 폴더 내부에 `cml.yml` 파일 생성했음
- 드래그해보고 화살표 제거해주면 됨(indent error 발생)
    - `git add .`
    - `git commit ~~`
    - `git push origin main`
- 위 명령어를 차례로 실행해주면 다음과 같이 생성됨

![Untitled](Chapter%203-%206aa3d/Untitled%201.png)

![Untitled](Chapter%203-%206aa3d/Untitled%202.png)

- 위와 같이 결과가 출력됨을 확인할 수 있음

---

### **CML 을 사용하여 성능 지표를 레포트 형태로 출력하기**

- [CML](https://cml.dev/) : 데이터 사이언스 프로젝트를 지속적으로 통합시키기 위한 오픈소스
    
    [GitHub - iterative/cml: ♾️ CML - Continuous Machine Learning | CI/CD for ML](https://github.com/iterative/cml)
    
- Markdown 형식으로 내보내기
    
    ```yaml
    echo "## MODEL METRICS" > report.md
    cat metrics.txt >> report.md
    ```
    
- CML Command 를 사용하여 분석 결과 이미지를 Markdown 형식으로 내보내기
    - [CML Command 살펴보기](https://cml.dev/doc/ref/pr)
    
    ```yaml
    echo "## Data viz" >> report.md
    cml-publish feature_importance.png --md >> report.md
    cml-publish residuals.png --md >> report.md
    cml-send-comment report.md
    ```
    
- Build

---

- 그럼 레포트는 어떻게 확인할 수 있을까?
- 아래 사진의 `오른쪽위`에 `commit`을 누름

![Untitled](Chapter%203-%206aa3d/Untitled%203.png)

![Untitled](Chapter%203-%206aa3d/Untitled%204.png)

![Untitled](Chapter%203-%206aa3d/Untitled%205.png)

### **분석 코드 변경 후 재배포 시 레포트 재생성 하기**

- 분석 코드 변경
    - max_depth 를 2 → 5 변경
    - commit 변경
    - Build
- 결과 변화 확인
- Commit 기록으로 해당 변화 시의 데이터,코드,환경 확인 가능

→ 위 방법을 `Commit message`로 구분하여 버져닝도 할 수 있음

---

### **DVC 를 활용하여 Metric 의 변화 추적하기**

- [Github Repo](https://github.com/yunjjun/github-actions-project/tree/github-actions-dvc) Fork
    - [데이터 확인](https://www.sciencedirect.com/science/article/pii/S2352340920303048) 및 다운로드
    - process_data.py 실행
    - train.py 실행
        
        ```bash
        ## AttributeError: 'Series' object has no attribute 'to_numpy' 에러 시 참고
        pip install --upgrade pandas
        ```
        
- Metric 의 변화는 어떻게 알 수 있을까?
    - [Git diff](https://git-scm.com/docs/git-diff)
        - 수치 비교보다는 코드의 변화를 알기 위해 보통 사용하므로 활용이 어렵다.
        - 파일 변경 이력을 알기 어렵다 → 모델링 전략 변경을 알기 어렵다.
    - [DVC](https://dvc.org/) 사용!
        - install dvc
        
        ```bash
        pip install dvc
        ```
        
        - 파이프라인 빌드
            - 초기화
                
                ```bash
                # 초기화
                dvc init
                # dvc.yaml 생성
                dvc run -n process -d process_data.py -d data_raw.csv -o data_processed.csv --no-exec python process_data.py
                ```
                
            
            - 위 코드 실행하면 다음과 같은 `yml` 파일 생성
            
            ```yaml
            stages:
              process:
                cmd: python process_data.py
                deps:
                - data_raw.csv
                - process_data.py
                outs:
                - data_processed.csv
            ```
            
            - 추가적으로 덧붙여서 아래에 `train` 내용을 추가해줌
            
            - 전체 코드
                
                ```yaml
                stages:
                  process:
                    cmd: python process_data.py
                    deps:
                    - process_data.py
                    - data_raw.csv
                    outs:
                    - data_processed.csv
                  train:
                    cmd: python train.py
                    deps:
                    - train.py
                    - data_processed.csv
                    outs:
                    - by_region.png
                    metrics:
                    - metrics.json:
                        cache: false
                ```
                
        - [파이프라인에 따라 재생성](https://dvc.org/doc/command-reference/repro)
            
            ```bash
            # 각 단계별로 재생성
            dvc repro
            ```
            
            - 터미널에 위 코드 실행하면
            
            ![Untitled](Chapter%203-%206aa3d/Untitled%206.png)
            
            - 전처리된 csv 및 시각화 파일들이 생성됨
            
            → 재생성될 때마다 `dvc repro` 실행하면 새로 적용됨
            
        - .github/workflows/train.yaml 생성
            
            ```yaml
            name: dvc-cml
            on: [push]
            jobs:
              run:
                runs-on: [ubuntu-latest]
                container: docker://dvcorg/cml-py3:latest
                steps:
                  - uses: actions/checkout@v2
                  - name: cml_run
                    env:
                      repo_token: ${{ secrets.GITHUB_TOKEN }}
                    run: |
                      pip install -r requirements.txt
                      dvc repro 
            
                      git fetch --prune ## https://git-scm.com/docs/git-fetch
                      dvc metrics diff --show-md master > report.md  # 마스터 기준으로 다른점 보여달라는 뜻
            
                      echo "## Validating results by region"
                      cml-publish by_region.png --md >> report.md
                      cml-send-comment report.md
            ```
            
        - 모델링 변경하여 테스트
            - 새로운 branch (experiment) 생성
            - [train.py](http://train.py) 수정 (LogisticRegression → [QuadraticDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html))
                
                ```python
                # from sklearn.linear_model import LogisticRegression
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                ...
                # clf = LogisticRegression()
                clf = QuadraticDiscriminantAnalysis()
                ```
                
            - commit → build
            - 변경값 확인 !
            
            ![Untitled](Chapter%203-%206aa3d/Untitled%207.png)