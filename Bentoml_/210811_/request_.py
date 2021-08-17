
import requests

file_path = './2.PNG'

with open(file_path, "rb") as f:
    image_bytes = f.read()

files = {
    "image": ("2nd_image", image_bytes),
}
response = requests.post("http://127.0.0.1:5000/predict", files=files)

print(response.text)