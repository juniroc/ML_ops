from fastapi import FastAPI

app = FastAPI()


@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    return {"file_path": lines}
