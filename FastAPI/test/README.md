
`main.py`

```
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
```

cmd : `python3 -m uvicorn main:app --reload` 

<img width="712" alt="image" src="https://user-images.githubusercontent.com/58424182/155266839-4f30c4f9-bdb2-40ff-b38d-e352f6ac58a8.png">


- 데코레이터에서 CRUD 사용 가능
  - `POST` : 데이터를 생성하기 위해
  - `GET` : 데이터를 읽기 위해
  - `PUT` : 데이터를 업데이트하기 위해
  - `DELETE` : 데이터를 삭제하기 위해

```
@app.post()
@app.put()
@app.delete()
```

- 경로 : `/` 을 기준으로 나뉨
  - "/" 만 입력했을 경우 `localhost:port`
  - "/test/1" : `localhost:port/test/1` 

- `async def` 대신 일반함수 `def` 도 가능

- return 값
  - dict, list, (단일 값)str, (단일 값)int 등 반환 가능

