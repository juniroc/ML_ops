# 5. Managing Research Code

![5%20Managing%20b5ec5/Untitled.png](5%20Managing%20b5ec5/Untitled.png)

![5%20Managing%20b5ec5/Untitled%201.png](5%20Managing%20b5ec5/Untitled%201.png)

### 리서치 코드 품질 문제

![5%20Managing%20b5ec5/Untitled%202.png](5%20Managing%20b5ec5/Untitled%202.png)

- ctrl-c + ctrl-v 기반으로 작성된 경우가 많음.

![5%20Managing%20b5ec5/Untitled%203.png](5%20Managing%20b5ec5/Untitled%203.png)

![5%20Managing%20b5ec5/Untitled%204.png](5%20Managing%20b5ec5/Untitled%204.png)

![5%20Managing%20b5ec5/Untitled%205.png](5%20Managing%20b5ec5/Untitled%205.png)

![5%20Managing%20b5ec5/Untitled%206.png](5%20Managing%20b5ec5/Untitled%206.png)

![5%20Managing%20b5ec5/Untitled%207.png](5%20Managing%20b5ec5/Untitled%207.png)

![5%20Managing%20b5ec5/Untitled%208.png](5%20Managing%20b5ec5/Untitled%208.png)

![5%20Managing%20b5ec5/Untitled%209.png](5%20Managing%20b5ec5/Untitled%209.png)

- import가 너무 꼬여있으면 디버깅이 어려움

![5%20Managing%20b5ec5/Untitled%2010.png](5%20Managing%20b5ec5/Untitled%2010.png)

### 린트와 유닛 테스트

![5%20Managing%20b5ec5/Untitled%2011.png](5%20Managing%20b5ec5/Untitled%2011.png)

![5%20Managing%20b5ec5/Untitled%2012.png](5%20Managing%20b5ec5/Untitled%2012.png)

![5%20Managing%20b5ec5/Untitled%2013.png](5%20Managing%20b5ec5/Untitled%2013.png)

![5%20Managing%20b5ec5/Untitled%2014.png](5%20Managing%20b5ec5/Untitled%2014.png)

![5%20Managing%20b5ec5/Untitled%2015.png](5%20Managing%20b5ec5/Untitled%2015.png)

![5%20Managing%20b5ec5/Untitled%2016.png](5%20Managing%20b5ec5/Untitled%2016.png)

### 린트

![5%20Managing%20b5ec5/Untitled%2017.png](5%20Managing%20b5ec5/Untitled%2017.png)

### Python 타입 체크 라이브러리

![5%20Managing%20b5ec5/Untitled%2018.png](5%20Managing%20b5ec5/Untitled%2018.png)

### 타입

![5%20Managing%20b5ec5/Untitled%2019.png](5%20Managing%20b5ec5/Untitled%2019.png)

![5%20Managing%20b5ec5/Untitled%2020.png](5%20Managing%20b5ec5/Untitled%2020.png)

![5%20Managing%20b5ec5/Untitled%2021.png](5%20Managing%20b5ec5/Untitled%2021.png)

![5%20Managing%20b5ec5/Untitled%2022.png](5%20Managing%20b5ec5/Untitled%2022.png)

![5%20Managing%20b5ec5/Untitled%2023.png](5%20Managing%20b5ec5/Untitled%2023.png)

### 타입 힌트

![5%20Managing%20b5ec5/Untitled%2024.png](5%20Managing%20b5ec5/Untitled%2024.png)

![5%20Managing%20b5ec5/Untitled%2025.png](5%20Managing%20b5ec5/Untitled%2025.png)

![5%20Managing%20b5ec5/Untitled%2026.png](5%20Managing%20b5ec5/Untitled%2026.png)

- 들어와야할 파라미터의 타입을 지정해줄 수 있음.(다른 사람이 파악하기 쉽게)

![5%20Managing%20b5ec5/Untitled%2027.png](5%20Managing%20b5ec5/Untitled%2027.png)

## CI(Continuous Integration) - 지속적 통합

![5%20Managing%20b5ec5/Untitled%2028.png](5%20Managing%20b5ec5/Untitled%2028.png)

- CI 에서는 기본적인 것 만 잡아주어도 큰 효과낼 수 있음.

---

## 실습

[chris-chris/research-ci-tutorial](https://github.com/chris-chris/research-ci-tutorial)

- 참고 github

### 1. Black 을 이용해 파이썬 코드 정리

```python
### requirements.txt 파일

black==19.10b0
coverage==4.4.1
codeclimate-test-reporter==0.2.3

### pip install -r requirements.txt 이용해서 다운로드
```

```python
### main.py 파일

def      helloworld(a):
    print(f"hello world! {a}!")#hmm..

if __name__ ==    "__main__":
    helloworld("nujnim")
```

- 위와 같이 정리가 안된 파이썬 파일 존재
→ 터미널에서 black [main.py](http://main.py) 를 이용하면

```bash

black [main.py](http://main.py)
```

```python
def helloworld(a):
    print(f"hello world! {a}!") #hmm..

if __name__ == "__main__":
    helloworld("nujnim")
```

- 위와 같이 정리됨

### 2. lint.yml 이용

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
→ 우분투 파이썬 3.7 에서 코드 체크 아웃하고 black 과 flake8 을 실행 (True가 표시된 경우)

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

![5%20Managing%20b5ec5/Untitled%2029.png](5%20Managing%20b5ec5/Untitled%2029.png)

- 위와 같이 CI 가 진행되고 있음을 알 수 있음.

![5%20Managing%20b5ec5/Untitled%2030.png](5%20Managing%20b5ec5/Untitled%2030.png)

- 자동으로 이미지 풀링

![5%20Managing%20b5ec5/Untitled%2031.png](5%20Managing%20b5ec5/Untitled%2031.png)

- 에러 발생
→ 코드 품질이 떨어지는 것을 확인

![5%20Managing%20b5ec5/Untitled%2032.png](5%20Managing%20b5ec5/Untitled%2032.png)

- 73~74 코드가 75~80 코드로 변환되어야 함을 알림

![5%20Managing%20b5ec5/Untitled%2033.png](5%20Managing%20b5ec5/Untitled%2033.png)

- Push Request를 막음
→ 원래는 막아야하나..

![5%20Managing%20b5ec5/Untitled%2034.png](5%20Managing%20b5ec5/Untitled%2034.png)

- 여기서는 그냥 Merge가 됨
→ Setting → Branches 에서 설정 바꾸기

![5%20Managing%20b5ec5/Untitled%2035.png](5%20Managing%20b5ec5/Untitled%2035.png)

![5%20Managing%20b5ec5/Untitled%2036.png](5%20Managing%20b5ec5/Untitled%2036.png)

- Setting → Branches → Branch protection rules(add_rule) 위와 같이 이름 지정하고

※ **유료 버전 아니면 못함**

---

### Unit_test 파일

```python
# main_test.py 파일

import unittest

import main

class MainTest(unittest.TestCase):
    def test_helloworld(self):
        ret = main.helloworld("test")
        self.assertEqual(ret, "hello world! nujnim!")

if __name__ == "__main__":
    unittest.main()
```

- 또한 Unit_test 가 통과하지 않으면 들어가지 않도록 해야함.

### Coverage.yml

- 테스트가 안되었을 때 통과가 안되도록 만들기

```yaml
# ./github/workflows/coverage.yml 파일

name: Coverage Report

# Run this workflow every time a new commit pushed to your repository
on: push

jobs:
  coverage-report:
    name: Coverage Report
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up Python 3.7
        uses: actions/setup-python@v2
        with:
          # Semantic version range syntax or exact version of a Python version
          python-version: '3.7' 
          # Optional - x64 or x86 architecture, defaults to x64
          architecture: 'x64'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      
			### 유닛 테스트 이후 coverage report 까지 뽑아내는 명령어
			- name: Generate coverage report
        run: |
          coverage run --source=./ -m unittest discover -p "*_test.py"
          coverage xml
      
      - name: Upload coverage to code climate
        env:
          CODECLIMATE_REPO_TOKEN: 51642184e2902baab3007c428a39a553590c79209aa133eb7fc093e3b021b775
        run: |
          codeclimate-test-reporter
```

- 유닛 테스트 이후 coverage report까지 뽑아냄

```python
git add.

git status

git commit -a -m "add unittest"

git push
```

![5%20Managing%20b5ec5/Untitled%2037.png](5%20Managing%20b5ec5/Untitled%2037.png)

- 실패해서 막힌 것을 확인.
→ 원래는 report가 따로 생성되어야하나 path 설정 안됨.

---

### Code climate (유료 툴)

- 로그인
→ 원하는 Repository _ Add Repo
→ Repo Settings → Test_coverage
    
    ![5%20Managing%20b5ec5/Untitled%2038.png](5%20Managing%20b5ec5/Untitled%2038.png)
    
- 토큰 복사 후 
coverage.yml 파일의 토큰 부분에 붙여넣기