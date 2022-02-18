# Chapter 3-2. Github Actions

![Untitled](Chapter%203-%208f4c8/Untitled.png)

- `Github` 로그인 → `Repository` 생성 → `Actions`

![Untitled](Chapter%203-%208f4c8/Untitled%201.png)

- `Python` 검색 후 `Python application` 선택

```yaml
# This workflow will install Python dependencies, run tests and lint with a single version of Python
# For more information see: https://help.github.com/actions/language-and-framework-guides/using-python-with-github-actions

name: Python application

on:
  push:
    branches: [ main ]    ## main branch 에서 push 일어나면 workflow 실행
  pull_request:
    branches: [ main ]

jobs:    ## 여기서 job 은 build
  build:
    runs-on: ubuntu-latest

    steps:    ## 다음 스텝으로 이 build 라는 작업을 실행해 달라는 뜻 (- 단위로 나눠짐)
    - uses: actions/checkout@v2    ## github.com 의 actions 라는 곳에 가면 나옴
    - name: Set up Python 3.10
      uses: actions/setup-python@v2  ## 파이썬 셋업하는 항목을 정의해둔 곳
      with:
        python-version: "3.10"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    - name: Test with pytest
      run: |
        pytest
```

- 위와 같은 `Yaml` 파일이 생성됨
→ 아래에 `Yaml` 내용 분석한 것 참고
- `Optional` / `Required` 확인
    - `on` : 조건을 나타냄
    - `job` : 한개 또는 여러개 가능(어떤 작업을 할 것인지)

`on` : main branch 에서 push 가 일어나면 workflow 실행

---

**Github Actions 를 활용하여 Docker Image 를 Build**

- Github 에 접속하여 새로운 Repository (github-actions-project) 생성
- Actions 클릭
    - 매우 다양한 툴들을 통합하기 쉽게 되어 있음
- Python Application 선택
- Workflow file 생성
    - yml 내용을 vscode 로 가져오기
    - Github event 알아보기
        - [https://docs.github.com/en/actions/learn-github-actions/events-that-trigger-workflows](https://docs.github.com/en/actions/learn-github-actions/events-that-trigger-workflows)
        
        ```yaml
        name [optional]
        on [required] events
        	: workflow 를 시작하게 할 수 있는 Github event 의 이름
        	: jobs [required]   jobs.<job_id>
        	: one or more jobs 
        	: sequence of tasks (steps)
        	: steps 1) can run commands, 2) setup tasks 3) run an action
        		- uses : selects an action (actions/ 다음에는 재사용 가능 코드 위치)
        		- run  : runs a command-line command
        ```
        
        - `Optional` / `Required` 확인
            - `on` : 조건을 나타냄
            - `job` : 한개 또는 여러개 가능(어떤 작업을 할 것인지)
            
        
        ```yaml
        name: Python application
        
        on:
          push:
            branches: [ python-ci-workflow ] ## python-ci-workflow branch 에서 push 일어나면 workflow 를 실행해달라는 뜻
          pull_request:
            branches: [ python-ci-workflow ]
        
        jobs: # 
          build:
            steps:
            - uses: actions/checkout@v2
            - name: Set up Python
              uses: actions/setup-python@v2
              with:
                python-version: "3.8"
            - name: Display Python version
              run: python -c "import sys; print(sys.version)"
        ```
        
    
    - 위 yaml 을 복사해서 (기존 yaml 에서 jobs 부분만 바꾼다
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%202.png)
    
    - `python-ci-workflow` 라는 `branch` 생성
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%203.png)
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%204.png)
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%205.png)
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%206.png)
    
    - `runs-on` 이 빠져있어서 생긴 에러
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%207.png)
    
    - 해당 브랜치로 이동해서 `yaml` 파일 수정
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%208.png)
    
    ```yaml
    # This workflow will install Python dependencies, run tests and lint with a single version of Python
    # For more information see: https://help.github.com/actions/language-and-framework-guides/using-python-with-github-actions
    
    name: Python application
    
    on:
      push:
        branches: [ main ]
      pull_request:
        branches: [ main ]
        
    jobs: 
      build:
    	  runs-on: ubuntu-latest
        steps:
        - uses: actions/checkout@v2
        - name: Set up Python
          uses: actions/setup-python@v2
          with:
            python-version: "3.8"
        - name: Display Python version
          run: python -c "import sys; print(sys.version)"
    ```
    
    - `runs-on` : 어떤 운영체제에서 실행할 것인지
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%209.png)
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%2010.png)
    
    ![Untitled](Chapter%203-%208f4c8/Untitled%2011.png)
    
    - Python version
    
    ---
    
    - actions 알아보기
        - [https://github.com/actions](https://github.com/actions)
        아래 페이지 참고하기
        
        [https://github.com/actions/setup-python](https://github.com/actions/setup-python)
        
        - checkout - action.yaml
    - yml 파일 이름을 ci.yml 으로 변경
    - Start commit → Create a new branch.. → 이름을 'python-ci-workflow' 로 변경 → Create pull request
    - Details 클릭 → build
    - 이 코드들은 어디서 실행되는 걸까?
        - Github 에 의해 관리된다
        - Workflow 의 각 jobs 은 새로운 가상 환경에서 실행된다
- runs-on 은 실행되는 서버의 운영체제를 나타낸다.
    - ubuntu, Windows, Mac
    
    ```yaml
    jobs:
    	build:
    		runs-on: ubuntu-latest
    		strategy:
    			matrix:
    				os: [ubuntu-latest, windows-latest, macOS-latest]
    ```
    
- ci.yml 을 업데이트 한다.
    - 세 가지 운영 체제 모두에서 세 개의 빌드가 병렬로 실행된다.
    
    ```yaml
    name: Python application
    
    on:
      push:
        branches: [ python-ci-workflow ]
      pull_request:
        branches: [ python-ci-workflow ]
    
    jobs:
      build:
        runs-on: ${{ matrix.os }}
        strategy:
          matrix:
            os: [ubuntu-latest, macos-latest, windows-latest]
            python-version: ['3.6', '3.8']
            exclude:
              - os: macos-latest
                python-version: '3.8'
              - os: windows-latest
                python-version: '3.6'
    
        steps:
        - uses: actions/checkout@v2
        - name: Set up Python
          uses: actions/setup-python@v2
          with:
            python-version: ${{ matrix.python-version }}
        - name: Display Python version
          run: python -c "import sys; print(sys.version)"
    ```
    

---

### **생성한 Docker Image 를 Docker Hub 에 Push 한다.**

- Workflow 설명
    - 작성한 App 을 빌드한다
    - Docker 이미지로 생성한다
    - Docker Repository 로 추가한다
- Dockerfile 예제 가져오기
    - [https://docs.docker.com/language/python/build-images/](https://docs.docker.com/language/python/build-images/)
        - 위 링크 참고해서
        
        ![Untitled](Chapter%203-%208f4c8/Untitled%2012.png)
        
        - 다음과 같이 python 파일 및 코드 입력 `Flask` 코드 입력
        
        ![Untitled](Chapter%203-%208f4c8/Untitled%2013.png)
        
        - `requirements.txt` 도 생성해줌
        
        ![Untitled](Chapter%203-%208f4c8/Untitled%2014.png)
        
        ```docker
        # syntax=docker/dockerfile:1
        
        FROM python:3.8-slim-buster
        
        WORKDIR /app
        
        COPY requirements.txt requirements.txt
        RUN pip3 install -r requirements.txt
        
        COPY . .
        
        CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
        ```
        

- Docker Image 빌드
    - Docker hub 로 이동 ([https://hub.docker.com/](https://hub.docker.com/))
    - github-actions-app 라는 이름으로 repo 생성
    - 구글에서 docker build and push action 으로 검색 ([https://github.com/marketplace/actions/docker-build-push-action](https://github.com/marketplace/actions/docker-build-push-action))
        - Docker Hub 지원
        - Secrets 설정 필요
            - github - settings - Secrets - New
            - Name : DOCKER_USERNAME - yunjjun
            - Name : DOCKER_PASSWORD - dockerhub 에서 access token 생성
            → 도커 허브에서 토큰 생성
            
            ![Untitled](Chapter%203-%208f4c8/Untitled%2015.png)
            
            - 비밀번호의 경우 실제 비밀번호와 Access Tokens 모두 가능
            
            ![Untitled](Chapter%203-%208f4c8/Untitled%2016.png)
            
            `f105c3ac-2e25-4615-bf33-808d0fdd36d0`
            
    
    - 다양한 [Inputs](https://github.com/marketplace/actions/docker-build-push-action#inputs)
    - 추가할 Dockerfile
        
        ```yaml
          - name: Build & push Docker image
            uses: mr-smithers-excellent/docker-build-push@v5
        		with:
        		  image: lmj3502/github-actions-project
        			tags: v2, latest
        		  registry: docker.io
        		  username: ${{ secrets.DOCKER_USERNAME }}
        		  password: ${{ secrets.DOCKER_PASSWORD }}
        ```
        
    - [본인 Docker Hub 아이디]/github-actions-app 으로 image 변경
    - os 는 ubuntu-latest 만 남겨두기
- ci.yml 최종본

```yaml
name: Python application

on:
  push:
    branches: [ python-ci-workflow ]
  pull_request:
    branches: [ python-ci-workflow ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: "3.8"
    - name: Display Python version
      run: python -c "import sys; print(sys.version)"
    - name: Build & push Docker image
      uses: mr-smithers-excellent/docker-build-push@v5
      with:
        image: lmj3502/github-actions-project
        tags: v3, latest
        registry: docker.io
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
```