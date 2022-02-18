# MLJAR

[mljar/mljar-supervised](https://github.com/mljar/mljar-supervised)

- **white-box**
- 과거에는 하이퍼파라미터만이 auto-ml의 역할이었음.

**MLJAR 의 4가지 모드**

![MLJAR%20cea2b/Untitled.png](MLJAR%20cea2b/Untitled.png)

1. **Explain - 빠른 EDA
- 약간의 모델로 기본적 하이퍼 파라미터 학습을 진행.**

2. **Compete - 높은 성능(high-accuracy model)
- 높은 정확도를 갖는 모델을 구축
- 4시간 이상의 학습 가능
- feature engineering, ensembling, stacking 방식 적용**

3. **Perform (학습 속도와 정확도 밸런스)**

4. **Optuna (시간 무제한 인 경우)
 - highly-tuned ML models**

### Feature Engineering

![MLJAR%20cea2b/Untitled%201.png](MLJAR%20cea2b/Untitled%201.png)

## Dacon 대회 데이터로 돌려보기

![MLJAR%20cea2b/Untitled%202.png](MLJAR%20cea2b/Untitled%202.png)

```python
import pandas as pd 
# scikit learn utilites
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# mljar-supervised package
from supervised.automl import AutoML

# load the data
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[1:-1]], df["credit"], test_size=0.2
)

# train models with AutoML
automl = AutoML(algorithms=["CatBoost", "Xgboost", "LightGBM"],
    model_time_limit=30*60,
    start_random_models=10,
    hill_climbing_steps=3,
    top_models_to_improve=3,
    golden_features=True,
    features_selection=True,
    stack_models=True,
    train_ensemble=True,
    explain_level=0,
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 4,
        "shuffle": False,
        "stratify": True,
    })
automl.fit(X_train, y_train)

# compute the accuracy on test data
predictions = automl.predict_all(X_test)
print(predictions.head())
print("Test accuracy:", accuracy_score(y_test, predictions["label"].astype(int)))
```

![MLJAR%20cea2b/Untitled%203.png](MLJAR%20cea2b/Untitled%203.png)

![MLJAR%20cea2b/Untitled%204.png](MLJAR%20cea2b/Untitled%204.png)

![MLJAR%20cea2b/Untitled%205.png](MLJAR%20cea2b/Untitled%205.png)

---

### 결과..

![MLJAR%20cea2b/Untitled%206.png](MLJAR%20cea2b/Untitled%206.png)

제출하였으나 50% 정도의 결과.

metric 설정하지 않았음을 감안하여도,

데이터가 편향되어있으며, 결측치가 존재하는데 

그것을 고려하지 못한 것도 있다.

하지만 "Explain" 모드로 빠르게 데이터를 EDA 해볼 수 있다는 점은 좋은 것 같다.

![MLJAR%20cea2b/Untitled%207.png](MLJAR%20cea2b/Untitled%207.png)

위 사진과 같이 Unique value들이 하나만 있을 때는 컬럼 자체를 제거해 버린다.

---

![MLJAR%20cea2b/Untitled%208.png](MLJAR%20cea2b/Untitled%208.png)

![MLJAR%20cea2b/Untitled%209.png](MLJAR%20cea2b/Untitled%209.png)

위 처럼 Missing value가 존재할 경우는 결측치를 중간값으로 처리

![MLJAR%20cea2b/Untitled%2010.png](MLJAR%20cea2b/Untitled%2010.png)

![MLJAR%20cea2b/Untitled%2011.png](MLJAR%20cea2b/Untitled%2011.png)

원핫인코딩이 아닌 멀티핫 인코딩멀티핫 인코딩

![MLJAR%20cea2b/Untitled%2012.png](MLJAR%20cea2b/Untitled%2012.png)

1. EDA 할땐 좋은듯
2. AutoML도 종류가 다양해 데이터에 맞는 모델 골라야함
3. 역시나 피쳐엔지니어링이 중요