### Python API 를 이용한 File Input

`request_.py`
```
import requests

file_path = './2.PNG'

with open(file_path, "rb") as f:
    image_bytes = f.read()

files = {
    "image": ("2nd_image", image_bytes),
}
response = requests.post("http://127.0.0.1:5000/predict", files=files)

print(response.text)

### "BAG"

```

**request_.py 실행**

```python request_.py``` \
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/210811_/capture/md_1.PNG)
