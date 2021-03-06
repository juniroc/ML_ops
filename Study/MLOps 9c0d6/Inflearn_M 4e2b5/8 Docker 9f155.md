# 8. Docker

![8%20Docker%209f155/Untitled.png](8%20Docker%209f155/Untitled.png)

![8%20Docker%209f155/Untitled%201.png](8%20Docker%209f155/Untitled%201.png)

![8%20Docker%209f155/Untitled%202.png](8%20Docker%209f155/Untitled%202.png)

![8%20Docker%209f155/Untitled%203.png](8%20Docker%209f155/Untitled%203.png)

![8%20Docker%209f155/Untitled%204.png](8%20Docker%209f155/Untitled%204.png)

![8%20Docker%209f155/Untitled%205.png](8%20Docker%209f155/Untitled%205.png)

![8%20Docker%209f155/Untitled%206.png](8%20Docker%209f155/Untitled%206.png)

![8%20Docker%209f155/Untitled%207.png](8%20Docker%209f155/Untitled%207.png)

![8%20Docker%209f155/Untitled%208.png](8%20Docker%209f155/Untitled%208.png)

![8%20Docker%209f155/Untitled%209.png](8%20Docker%209f155/Untitled%209.png)

![8%20Docker%209f155/Untitled%2010.png](8%20Docker%209f155/Untitled%2010.png)

![8%20Docker%209f155/Untitled%2011.png](8%20Docker%209f155/Untitled%2011.png)

![8%20Docker%209f155/Untitled%2012.png](8%20Docker%209f155/Untitled%2012.png)

- 도커 컨테이너를 만들기 위한 읽기 전용 템플릿

![8%20Docker%209f155/Untitled%2013.png](8%20Docker%209f155/Untitled%2013.png)

- 실제 실행 가능한 인스턴스 (CLI 이용)

![8%20Docker%209f155/Untitled%2014.png](8%20Docker%209f155/Untitled%2014.png)

![8%20Docker%209f155/Untitled%2015.png](8%20Docker%209f155/Untitled%2015.png)

```python
sudo docker run -d -p 80:80 docker/getting-started
```

- 도커 실행
→ [localhost:80](http://localhost:80) 으로 입장가능

- 도커 파일 생성

```python
# vi Dockerfile
# pwd: /workplace/Dockerfile

FROM node:12-alpine
RUN apk add --no-cache python g++ make
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
```

```python
### 도커파일 빌드
sudo docker build -t getting-started .
```

- .  : 현재 디렉토리에서 Dockerfile 을 찾으라는 뜻
- -t : 이미지 flag tags

![8%20Docker%209f155/Untitled%2016.png](8%20Docker%209f155/Untitled%2016.png)

![8%20Docker%209f155/Untitled%2017.png](8%20Docker%209f155/Untitled%2017.png)

![8%20Docker%209f155/Untitled%2018.png](8%20Docker%209f155/Untitled%2018.png)

![8%20Docker%209f155/Untitled%2019.png](8%20Docker%209f155/Untitled%2019.png)

![8%20Docker%209f155/Untitled%2020.png](8%20Docker%209f155/Untitled%2020.png)

![8%20Docker%209f155/Untitled%2021.png](8%20Docker%209f155/Untitled%2021.png)

![8%20Docker%209f155/Untitled%2022.png](8%20Docker%209f155/Untitled%2022.png)

![8%20Docker%209f155/Untitled%2023.png](8%20Docker%209f155/Untitled%2023.png)

![8%20Docker%209f155/Untitled%2024.png](8%20Docker%209f155/Untitled%2024.png)

![8%20Docker%209f155/Untitled%2025.png](8%20Docker%209f155/Untitled%2025.png)

![8%20Docker%209f155/Untitled%2026.png](8%20Docker%209f155/Untitled%2026.png)

![8%20Docker%209f155/Untitled%2027.png](8%20Docker%209f155/Untitled%2027.png)

![8%20Docker%209f155/Untitled%2028.png](8%20Docker%209f155/Untitled%2028.png)

![8%20Docker%209f155/Untitled%2029.png](8%20Docker%209f155/Untitled%2029.png)