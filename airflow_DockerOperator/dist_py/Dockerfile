# 베이스 이미지 불러오기
FROM python:3.8.12-slim

# 작업 디렉토리 설정
WORKDIR /gnn_py

# 현재 디렉토리 안의 모든 파일 작업 디렉토리로 이동
## python 파일 및 requirements
COPY . /gnn_py

# 권한 재설정
RUN chmod -R 755 /gnn_py

# 버전 디렉토리, Config 디렉토리 생성
RUN mkdir /gnn_py/versions
RUN mkdir /gnn_py/config_files
RUN mkdir /gnn_py/dataset

# vim 및 리눅스 패키지 인스톨
RUN apt update && apt install -y vim
RUN apt install -y procps


# 패키지 설치 & dgl 설치
RUN pip install --upgrade pip
RUN pip install dgl -f dgl -f https://data.dgl.ai/wheels/repo.html
RUN pip install -r requirements.txt

# KST로 시간대 변경
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
