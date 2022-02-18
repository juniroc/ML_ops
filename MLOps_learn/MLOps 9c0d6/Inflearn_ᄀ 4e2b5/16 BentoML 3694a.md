# 16. BentoML (16/16)

![16%20BentoML%203694a/Untitled.png](16%20BentoML%203694a/Untitled.png)

![16%20BentoML%203694a/Untitled%201.png](16%20BentoML%203694a/Untitled%201.png)

### bentoML

![16%20BentoML%203694a/Untitled%202.png](16%20BentoML%203694a/Untitled%202.png)

- serving에 집중

![16%20BentoML%203694a/Untitled%203.png](16%20BentoML%203694a/Untitled%203.png)

## STEP

![16%20BentoML%203694a/Untitled%204.png](16%20BentoML%203694a/Untitled%204.png)

설치

```python
pip install bentoml 
```

### Helloworld

`iris.py`

```python
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC(gamma='scale')
clf.fit(X,y)
```

- 위는 기본 모델
- 이런 모델을 패키징해서 배포하기 위한 방법을 다룸

---

### 하나의 벤토 패키징

`iris_classifier.py`

```python
import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)
```

- 위의 것들이 하나의 docker image 가 된다고 생각하면 됨

**단, 위 코드는 어떤 모델이 아닌 모델 껍데기 정도라고 생각하면 됨.(image로 패키징하기 위한)**

---

- **env → 서비스 환경자체에 대해 설정**
→ pip 패키지를 무엇으로 할지, pandas or numpy 등
→ 현재 개발하고있는 곳의 package들을 갖다 쓰는것을 `infer_pip_packages=True` 로 표현 가능
→ 사실 위 방법보다는 requirement.txt 또는 하나씩 지정해주는 것이 더 명확
- **artifacts → ML을 학습시키고 artifact로 저장함**
→ artifacts.model 을 입력하면 모델에 접근할 수 있음.
- **api → 예측서비스에 액세스하기위한 endpoint**

---

`iris_service.py`

- 이 코드를 이용해 iris_classifier 에서 정의한 예측 서비스 클래스로 학습 된 모델을 패키징
→ 이후 배포를 위해 iris_classifier 인스턴스를 bentoML 형식으로 디스크에 저장

```python
# import the IrisClassifier class defined above
from iris_classifier import IrisClassifier  ## 이것을 로드해서 패키징
from sklearn import svm
from sklearn import datasets

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

### svm 모델이 irisClassifier와 'model' 로 패키징됨
# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)

## 경로를 저장해줌
# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()
```

![16%20BentoML%203694a/Untitled%205.png](16%20BentoML%203694a/Untitled%205.png)

![16%20BentoML%203694a/Untitled%206.png](16%20BentoML%203694a/Untitled%206.png)

### `iris_service.py` 실행

```bash
python iris_service.py
```

![16%20BentoML%203694a/Untitled%207.png](16%20BentoML%203694a/Untitled%207.png)

![16%20BentoML%203694a/Untitled%208.png](16%20BentoML%203694a/Untitled%208.png)

### server 실행

```bash
bentoml serve IrisClassifier:latest
```

- 여기서 IrisClassifier 는 **클래스 명** 그대로

![16%20BentoML%203694a/Untitled%209.png](16%20BentoML%203694a/Untitled%209.png)

![16%20BentoML%203694a/Untitled%2010.png](16%20BentoML%203694a/Untitled%2010.png)

---

### cli_inference 방법

```bash
bentoml run IrisClassifier:latest predict --input "[[5.1, 3.5, 1.4, 0.2]]"
```

![16%20BentoML%203694a/Untitled%2011.png](16%20BentoML%203694a/Untitled%2011.png)

- 아래에 [0] 이라고 분류한 것을 알 수 있음

### curl_inference 방법

```bash
curl -i -H "Content-Type: application/json" --request POST --data "[[5.1, 3.5, 1.4, 0.2
]]" http://127.0.0.1:5000/predict
```

![16%20BentoML%203694a/Untitled%2012.png](16%20BentoML%203694a/Untitled%2012.png)

- 이것 또한 사진 처럼 결과 나옴

### IrisClassifier 를  Containerize

```bash
bentoml containerize IrisClassifier:latest -t iris-classifier
```

- 위 코드는 잘 안되는 것 같음

---

그래서 Document 찾아서 아래와 같은 방법으로 container 생성

```bash
# Find the local path of the latest version IrisClassifier saved bundle
saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)

# Build docker image using saved_path directory as the build context, replace the
# {username} below to your docker hub account name
sudo docker build -t {username}/iris_classifier_bento_service $saved_path

# Run a container with the docker image built and expose port 5000
sudo docker run -p 5000:5000 {username}/iris_classifier_bento_service

### 문서에는 없지만 docker에 login 해주어야함
sudo docker login

# Push the docker image to docker hub for deployment
sudo docker push {username}/iris_classifier_bento_service
```

![16%20BentoML%203694a/Untitled%2013.png](16%20BentoML%203694a/Untitled%2013.png)

![16%20BentoML%203694a/Untitled%2014.png](16%20BentoML%203694a/Untitled%2014.png)

![16%20BentoML%203694a/Untitled%2015.png](16%20BentoML%203694a/Untitled%2015.png)

---

### 적응형 마이크로 배칭

![16%20BentoML%203694a/Untitled%2016.png](16%20BentoML%203694a/Untitled%2016.png)

![16%20BentoML%203694a/Untitled%2017.png](16%20BentoML%203694a/Untitled%2017.png)

![16%20BentoML%203694a/Untitled%2018.png](16%20BentoML%203694a/Untitled%2018.png)

![16%20BentoML%203694a/Untitled%2019.png](16%20BentoML%203694a/Untitled%2019.png)

![16%20BentoML%203694a/Untitled%2020.png](16%20BentoML%203694a/Untitled%2020.png)

![16%20BentoML%203694a/Untitled%2021.png](16%20BentoML%203694a/Untitled%2021.png)

![16%20BentoML%203694a/Untitled%2022.png](16%20BentoML%203694a/Untitled%2022.png)

![16%20BentoML%203694a/Untitled%2023.png](16%20BentoML%203694a/Untitled%2023.png)

### practice

`iris_classifier.py`

- **기존**에서 **@api 데코레이터** 아래 **파라미터 정의**하는 부분만 바뀜

```python
import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class IrisMicroBatchClassifier(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @api(
        input=DataframeInput(),
        mb_max_latency=10000, 
        mb_max_batch_size=1000,
        batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)
```

### Fashion MNIST 모델 서빙 Practice

`train.py`

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import io

# TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(_train_images, train_labels), (_test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = _train_images / 255.0
test_images = _test_images / 255.0

class FashionMnist(tf.keras.Model):
    def __init__(self):
        super(FashionMnist, self).__init__()
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    @staticmethod
    def image_bytes2tensor(inputs):
        with tf.device("cpu:0"):  # map_fn has issues on GPU https://github.com/tensorflow/tensorflow/issues/28007
            inputs = tf.map_fn(lambda i: tf.io.decode_png(i, channels=1), inputs, dtype=tf.uint8)
        inputs = tf.cast(inputs, tf.float32)
        inputs = (255.0 - inputs) / 255.0
        inputs = tf.reshape(inputs, [-1, 28, 28])
        return inputs

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)])
    def predict_image(self, inputs):
        inputs = self.image_bytes2tensor(inputs)
        return self(inputs)
    
    def call(self, inputs):
        return self.cnn(inputs)

# pick up a test image
d_test_img = _test_images[0]
print(class_names[test_labels[0]])

plt.imshow(255.0 - d_test_img, cmap='gray')
plt.imsave("test.png", 255.0 - d_test_img, cmap='gray')

# read bytes
with open("test.png", "rb") as f:
    img_bytes = f.read()

# verify saved image
assert tf.reduce_mean(FashionMnist.image_bytes2tensor(tf.constant([img_bytes])) - d_test_img) < 0.01

model = FashionMnist()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=50)

predict = model.predict_image(tf.constant([img_bytes]))
klass = tf.argmax(predict, axis=1)
[class_names[c] for c in klass]

from tensorflow_fashion_mnist import FashionMnistTensorflow

bento_svc = FashionMnistTensorflow()
bento_svc.pack("model", model)
saved_path = bento_svc.save()
```

`tensorflow_fashion_mnist.py`

```python
import bentoml
import tensorflow as tf

from bentoml.artifact import TensorflowSavedModelArtifact
from bentoml.adapters import TfTensorInput

FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'pillow'])
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class FashionMnistTensorflow(bentoml.BentoService):

    @bentoml.api(input=TfTensorInput(), batch=True)
    def predict(self, inputs):
        outputs = self.artifacts.model.predict_image(inputs)
        output_classes = tf.math.argmax(outputs, axis=1)
        return [FASHION_MNIST_CLASSES[c] for c in output_classes]
```

```python
python train.py
```

### server 띄우기

```bash
bentoml serve FashionMnistTensorflow:latest
```

**아래 linux 명령어로 test.json 파일 생성** 

```bash
echo "{\"instances\":[{\"b64\" : \" $(base64 test.png) \" }]}" > test.json
```

![16%20BentoML%203694a/Untitled%2024.png](16%20BentoML%203694a/Untitled%2024.png)

```bash
cat test.json

>>
{"instances":[{"b64" : "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90
bGliIHZlcnNpb24zLjQuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8rg+JYAAAACXBIWXMAAA9h
AAAPYQGoP6dpAAADZ0lEQVR4nO2WT0vrShiHn0ynjdrEmiJWPYpuhK5c+glcuRD8XH4CF7px4Ubc
K11aUVwV/1EqWEVq2lJta5o0scmchbc5Hu6By70HPFxwYCBkJu/zzu/9vUk0pZTiE4f4TNgX8Av4
54FhGPKxy3zfB6BSqfweUClFFEUAPD4+sr+/j+M4JBIJNE2L9+m6DsDBwcHvAQGEeH/0+PiYYrHI
zs7O3/Y0Gg12d3cxTTO+J/8LLAxDpJScn59zc3NDLpejUqmwsbGBZVn0+30WFhZotVp0u13m5uZ+
JPpvYVEUIaXEcRz29/dRStHv93l9fUUpFc+rqyuklFiWxWAw+GfgsPhRFMXXYRjGUm5tbZHL5Zie
nsbzPPr9PrlcLq5jOp1G13V836fb7eI4zq+Bw+DD4gsh0DSNMAxJJBIA7O3tYds209PTGIZBu90m
m80yNTVFMpkkDEPe3t7ieL1ej9vb218Dh6AoihgMBnECQ9jOzg6lUon5+XlarRbtdhvP88hkMry+
vqJpGmNjYySTSZRScbzDw0Pgg2mGNtc0DaUUQohYPoBarcbBwQGe57G0tITjOPi+T6vVIpVKoWka
ruvGyem6jhCCdDqNEIJisfgOHEr1Mfgwq2azSbVapVwu8/T0RCqVYnx8nHa7Tbfb5e3tDd/3EUJQ
rVYZDAZMTEyQTCYRQqCUYnR0lDAMMQyDy8tL5FCqer3O/f09vV6PXq+H53nc3d3hui5SSkzTJIoi
Op0OnuchpcR1XUZHR9F1nSAImJ2dpdPp4LoulmXhOA4vLy+k02ls2+b5+fld0kKhQK1WQ0pJs9mM
DTIEOY6DbdsopfB9H8uyiKIIx3EIw5B0Oo1hGGQyGRqNRqyUZVkIIfA8jyAIkFIij46O2N7eJp/P
MzMzE58klUrF70bTNAmCACFE3G+e56FpGlEUYds29Xqd6+trgiAgDEMADMPAdV10XccwDKamppAr
Kyucnp5ycXHxo7B/nSybzZLNZslkMgRBgFKKVqtFuVzGdV263S6aplEqlVheXmZxcZFCoYDv+7EP
pJR8+/YN0zTfXfzxJ8pxHM7OziiXy5ycnNBsNn9q2qEDs9ks+Xye1dVV1tbWGBkZidfX19d5eHhg
cnIS0zQxTRMpJbqus7m5+TPwM8b/+wP8BfwjwO92Cc7FhT8TpgAAAABJRU5ErkJggg=="}]}
```

![16%20BentoML%203694a/Untitled%2025.png](16%20BentoML%203694a/Untitled%2025.png)

### 생성된 json 파일 형태로 요청 - Inference

```bash
curl --request POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @test.json
```

![16%20BentoML%203694a/Untitled%2026.png](16%20BentoML%203694a/Untitled%2026.png)

![16%20BentoML%203694a/Untitled%2027.png](16%20BentoML%203694a/Untitled%2027.png)