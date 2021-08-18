# Model serving using Flask

1. Image를 byte 형태로 Request

2. 전처리 model_.transform_image()

3. 인퍼런스 model_.get_prediction()

4. json 형태로 return

5. request.py 를 통해 이미지 전달 (request.post) 및 인퍼런스 결과 출력 (request.json())

6. FLASK_ server띄우기
```
### app <- python file
export FLASK_APP=app 

flask run
```

![cap1](https://github.com/juniroc/ML_ops/blob/main/model_serving_using_flask/capture/cap_1.JPG)

![cap2](https://github.com/juniroc/ML_ops/blob/main/model_serving_using_flask/capture/cap_2.JPG)

