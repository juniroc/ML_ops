# 15. mlflow

![15%20mlflow%209fb94/Untitled.png](15%20mlflow%209fb94/Untitled.png)

![15%20mlflow%209fb94/Untitled%201.png](15%20mlflow%209fb94/Untitled%201.png)

- ML flow 가 왜 필요할까?

![15%20mlflow%209fb94/Untitled%202.png](15%20mlflow%209fb94/Untitled%202.png)

![15%20mlflow%209fb94/Untitled%203.png](15%20mlflow%209fb94/Untitled%203.png)

- Project - 어떤 플랫폼에서도 구현 가능하도록

![15%20mlflow%209fb94/Untitled%204.png](15%20mlflow%209fb94/Untitled%204.png)

### 간단실습 (실습예제의 수치는 의미 없음) 그냥 메트릭에 어떻게 입력되는지 확인

```python
import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")
```

![15%20mlflow%209fb94/Untitled%205.png](15%20mlflow%209fb94/Untitled%205.png)

### UI 로 log 확인

![15%20mlflow%209fb94/Untitled%206.png](15%20mlflow%209fb94/Untitled%206.png)

![15%20mlflow%209fb94/Untitled%207.png](15%20mlflow%209fb94/Untitled%207.png)

위와 같이 메트릭도 확인 가능

### sklearn_iris

```python
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)

# 입출력 정보를 정해주는 부분. 이런 정보를 시그니처라고 한다.
signature = infer_signature(iris_train, clf.predict(iris_train))

# 위에서 정한 시그니처 값을 인자로 넘긴다.
mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)
```

---

입출력 정보를 정해주는 부분. 이런 정보를 시그니처라고 한다.

위에서 정한 시그니처 값을 인자로 넘긴다.

---

여기서 `log_model()` 는 `save_model()` 과 같이 비슷함.
→ 저장되는 위치가 run 내부라는 점

그 결과 아래와 같이 모델들이 저장됨 (yaml파일 형식)

![15%20mlflow%209fb94/Untitled%208.png](15%20mlflow%209fb94/Untitled%208.png)

![15%20mlflow%209fb94/Untitled%209.png](15%20mlflow%209fb94/Untitled%209.png)

![15%20mlflow%209fb94/Untitled%2010.png](15%20mlflow%209fb94/Untitled%2010.png)

![15%20mlflow%209fb94/Untitled%2011.png](15%20mlflow%209fb94/Untitled%2011.png)

### I/O tpye을 정해줄 수 있음

```python
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)

input_schema = Schema([
    ColSpec("double", "sepal length (cm)"),
    ColSpec("double", "sepal width (cm)"),
    ColSpec("double", "petal length (cm)"),
    ColSpec("double", "petal width (cm)")
])

output_schema = Schema([ColSpec("long")])
signature =ModelSignature(inputs=input_schema, outputs=output_schema)

mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)
mlflow.sklearn.save_model(clf, "iris_srf")
```

![15%20mlflow%209fb94/Untitled%2012.png](15%20mlflow%209fb94/Untitled%2012.png)

**⇒**

![15%20mlflow%209fb94/Untitled%2013.png](15%20mlflow%209fb94/Untitled%2013.png)

이때 save_model 인경우는 iris_srf 폴더에 저장

![15%20mlflow%209fb94/Untitled%2014.png](15%20mlflow%209fb94/Untitled%2014.png)

sample_example

```python
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)

input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = {
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
mlflow.sklearn.log_model(clf, "iris_rf", signature=signature, input_example=input_example)
mlflow.sklearn.save_model(path="iris_rf", sk_model=clf)
```

다음과 같은 input_example.json 생성

![15%20mlflow%209fb94/Untitled%2015.png](15%20mlflow%209fb94/Untitled%2015.png)

### 모델 서빙

![15%20mlflow%209fb94/Untitled%2016.png](15%20mlflow%209fb94/Untitled%2016.png)

conda error 가 뜰 경우 뒤에 —no-conda 를 추가해주면됨

```python
curl --location --request POST 'localhost:1234/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "columns":["sepal length (cm)", "sepal width (cm)", "petal length (cm)",  "petal width (cm)"],
    "data": [[5.1, 3.5, 1.4, 0.2]]
}'
```

윈도우에서는 아래와 같이 변형시켜서 해야함.

![15%20mlflow%209fb94/Untitled%2017.png](15%20mlflow%209fb94/Untitled%2017.png)

```python
curl http://127.0.0.1:1234/invocations -H "Content-Type: application/json" --data "{\"columns\":[\"sepal length (cm)\", \"sepal width (cm)\", \"petal length (cm)\", \"petal width (cm)\"], \"data\":[[4.5, 3.5, 3.4, 1.2]]}"
```

---

### tf_mnist

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
trainX = train_X.reshape((train_X.shape[0], 28, 28, 1))
testX = test_X.reshape((test_X.shape[0], 28, 28, 1))
trainY = tf.keras.utils.to_categorical(train_Y)
testY = tf.keras.utils.to_categorical(test_Y)

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY))

signature = infer_signature(testX, model.predict(testX))
mlflow.keras.log_model(model, "mnist_cnn", signature=signature) 
```

### PyFunc 모델_Add N

- 커스텀 모델을 생성하고 싶은 경우 이용한다.
→ sklearn과 deeplearning 모델을 동시에 사용하고 싶은 경우
- pyfunc로 모델을 로드하여 서빙도 가능

```python
import mlflow.pyfunc

# Define the model class
class AddN(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)

# Construct and save the model
model_path = "add_n_model"
add5_model = AddN(n=5)
mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)

# Evaluate the model
import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)
assert model_output.equals(pd.DataFrame([range(5, 15)]))
```

**똑같이 mlflow 모델 서버 올려주고**

```python
mlflow models serve -m add_n_model -p 1234 --no-conda
```

![15%20mlflow%209fb94/Untitled%2018.png](15%20mlflow%209fb94/Untitled%2018.png)

**다른 터미널에서 숫자 서빙**

```python
curl http://127.0.0.1:1234/invocations -H "Content-Type: application/json" --data "[4]
```

---

### XGBoost_Iris

```python
# Load training and test datasets
from sys import version_info
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
iris = datasets.load_iris()
x = iris.data[:, :]
y = iris.target
x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(x_train, label=y_train)

# Train and save an XGBoost model
xgb_model = xgb.train(params={'max_depth': 10}, dtrain=dtrain, num_boost_round=10)
xgb_model_path = "xgb_model.pth"
xgb_model.save_model(xgb_model_path)

# Create an `artifacts` dictionary that assigns a unique name to the saved XGBoost model file.
# This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
# into the new MLflow Model's directory.
artifacts = {
    "xgb_model": xgb_model_path
}

# Define the model class
import mlflow.pyfunc
class XGBWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import xgboost as xgb
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["xgb_model"])

    def predict(self, context, model_input):
        input_matrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(input_matrix)

# Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'xgboost=={}'.format(xgb.__version__),
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'xgb_env'
}

# Save the MLflow Model
mlflow_pyfunc_model_path = "xgb_mlflow_pyfunc"
mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path, python_model=XGBWrapper(), artifacts=artifacts,
        conda_env=conda_env)

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# Evaluate the model
import pandas as pd
test_predictions = loaded_model.predict(pd.DataFrame(x_test))
print(test_predictions)
```

---

## Model registry

### mlflow_server.sh

```python
mlflow server --backend-store-uri sqlite.db --default-artifac
t-root ~/mlflow
```

모델 등록

### mlflow_host.sh

```python
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

**-> 여기서 우리는 cmd 이므로**

set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### random_forest.py

```python
from random import random, randint
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="YOUR_RUN_NAME") as run:
    params = {"n_estimators": 5, "random_state": 42}
    sk_learn_rfr = RandomForestRegressor(**params)

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    mlflow.log_param("param_1", randint(0, 100))
    mlflow.log_metrics({"metric_1": random(), "metric_2": random() + 1})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=sk_learn_rfr,
        artifact_path="sklearn-model",
        registered_model_name="sk-learn-random-forest-reg-model"
    )
```

위와 같은 파이썬 파일 (모델 생성 후 register, 학습은 진행되지 않음) 생성하여 실행

```python
python random_forest.py
```