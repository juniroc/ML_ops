import requests

resp = requests.post("http://localhost:5000/predict",
                    files={"file": open('./file.JPG','rb')})

print(resp.json())
