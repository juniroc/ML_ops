# 1-1. Data and Model Management(DVC)

# DVC

![Untitled](1-1%20Data%20a%20fa836/Untitled.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%201.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%202.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%203.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%204.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%205.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%206.png)

- Data Version Control

![Untitled](1-1%20Data%20a%20fa836/Untitled%207.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%208.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%209.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2010.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2011.png)

```bash
sudo apt install git

git --version
# git version 2.25.1

git --help
# 정상 설치되었는지 확인
```

- dvc 설치
    - `dvc[all]` 에서 `[all]` 은 dvc 의 remote storage 로 s3, gs, azure, oss, ssh 모두를 사용할 수 있도록 관련 패키지를 함께 설치하는 옵션

```bash
pip install dvc[all]==2.6.4

dvc --version

dvc --help
```

### DVC 저장소 셋팅

1. 새 directory 생성 및 **git 저장소로 초기화**

```bash
mkdir dvc-tutorial

cd dvc-tutorial

# git 저장소로 초기화
git init
```

1. 해당 Directory를 dvc 저장소로 초기화

```bash
dvc init
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2012.png)

### DVC 기본 명령 1

1. dvc로 버전 tracking 할 data 생성

```bash
# data 를 저장할 용도로 data 라는 이름의 디렉토리를 생성하고 이동
mkdir data

cd data

# 가볍게 변경할 수 있는 데이터를 카피해오거나, 새로 만듦
vi demo.txt

cat demo.txt
# Hello Fast campus!!!
```

1. 위에서 생성한 데이터를 `dvc`로 `tracking`

```bash
cd ..

dvc add data/demo.txt

# To track the changes with git, run:
git add data/demo.txt.dvc data/.gitignore
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2013.png)

1. `dvc` `add` 에 의해 자동 생성된 파일 확인

```bash
cd data

ls
# demo.txt.dvc 파일이 생성된 것을 확인

cat demo.txt.dvc
# demo.txt 파일의 메타정보 파일
# git 에서는 demo.txt 파일이 아닌, demo.txt.dvc 파일만 관리
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2014.png)

1. git commit 수행

```bash
git commit -m "Add demo.txt.dvc"
```

- (`.dvc` 파일은 `git push` 를 수행하여, `git repository` 에 저장

![Untitled](1-1%20Data%20a%20fa836/Untitled%2015.png)

1. data 가 실제로 저장될 remote storage 를 세팅
- 본인의 google drive 에 새로운 폴더를 하나 생성한 뒤, url로 부터 ID를 복사
    - 아래 사진의 빨간 네모 부분

![Untitled](1-1%20Data%20a%20fa836/Untitled%2016.png)

[`https://drive.google.com/drive/folders/1VW6i-FEdUVwP8yu1GI3VUIlK8nU5Ui5p`](https://drive.google.com/drive/folders/1VW6i-FEdUVwP8yu1GI3VUIlK8nU5Ui5p)

- 이건 내꺼

[`1VW6i-FEdUVwP8yu1GI3VUIlK8nU5Ui5p`](https://drive.google.com/drive/folders/1VW6i-FEdUVwP8yu1GI3VUIlK8nU5Ui5p)

- 이부분 가져다 쓰면됨.

```bash
dvc remote add -d storage gdrive://<Google_drive_folder_id>
# dvc 의 default remote storage 로 gdrive://<Google_drive_folder_id> 세팅
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2017.png)

1. `dvc` `config`를 `git` `commit` 

```bash
git add .dvc/config

git commit -m "add: add remote storage"
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2018.png)

1. dvc push
- 데이터를 remote storage 에 업로드

```bash
dvc push
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2019.png)

- `authentication` 링크를 통해서 입력 후 push

- drive 들어가서 확인

![Untitled](1-1%20Data%20a%20fa836/Untitled%2020.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2021.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2022.png)

- 기존 생성했던 파일과 같은 파일임을 확인할 수 있음

## DVC 기본 명령 2.

1. dvc pull
    - 데이터를 remote storage 로부터 다운로드

```bash
rm -rf .dvc/cache/
# 캐쉬 삭제

rm -rf data/demo.txt
# dvc push 했던 데이터 삭제

dvc pull
# dvc pull 로 google drive에 업로드했던 데이터 다운
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2023.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2024.png)

1. `dvc checkout`
    - data 의 버전 변경하는 명령어
    - 버전 변경 테스트를 위해, `새로운 버전`의 `data`를 `dvc push`
    
    ```bash
    vi demo.txt
    # 들어가서 데이터 내용 수정
    
    dvc add data/demo.txt
    # dvc add (data/demo.txt.dvc 를 변경시켜주는 역할)
    
    git add data/demo.txt.dvc
    git commit -m "update: update demo.txt"
    # git add and commit
    
    dvc push
    
    git push
    # dvc push (and git push) 새로운 버전의 data 파일을 remote storage 에 업로드
    ```
    

![Untitled](1-1%20Data%20a%20fa836/Untitled%2025.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2026.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2027.png)

```bash
git log --oneline
# git log 확인

git checkout <Commit_hash> data/demo.txt.dvc
# demo.txt.dvc 파일을 이전 commit 버전으로 되돌림

dvc checkout
# dvc checkout (demo.txt.dvc 의 내용을 보고 demo.txt 파일을 이전 버전으로 변경)

cat data/demo.txt
# 데이터 변경 확인
```

![Untitled](1-1%20Data%20a%20fa836/Untitled%2028.png)

![Untitled](1-1%20Data%20a%20fa836/Untitled%2029.png)

### DVC 추가 기능

![Untitled](1-1%20Data%20a%20fa836/Untitled%2030.png)

 

---