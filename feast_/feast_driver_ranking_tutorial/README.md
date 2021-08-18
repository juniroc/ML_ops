### 0. Colab 이용

---

### 1. Install libraries

```
pip install feast scikit-learn parquet-cli
```

---

### 2. Initialize Repository
* 원하는 이름으로 feast 초기 폴더를 설정한다.
* `feast_driver_ranking_tutorial`

```
feast init feast_driver_ranking_tutorial
```

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/1.PNG)

---

### 3. feast apply
* init으로 생성된 yaml 파일과 python 파일을 토대로 feature_store 에 등록

`example.py`
```
# This is an example feature definition file

from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
driver_hourly_stats = FileSource(
    path="/content/feast_driver_ranking_tutorial/data/driver_stats.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the driver. You can think of entity as a primary key used to
# fetch features.
driver = Entity(name="driver_id", value_type=ValueType.INT64, description="driver id",)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
driver_hourly_stats_view = FeatureView(
    name="driver_hourly_stats",
    entities=["driver_id"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="conv_rate", dtype=ValueType.FLOAT),
        Feature(name="acc_rate", dtype=ValueType.FLOAT),
        Feature(name="avg_daily_trips", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=driver_hourly_stats,
    tags={},
)

```

`feature_store.yaml`
```
project: feast_driver_ranking_tutorial
registry: data/registry.db
provider: local
online_store:
    path: data/online_store.db
```



![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/2.PNG)


`data_list`

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/6.PNG)

---

### 4. 깃에서 driver_orders.csv 파일을 가져온다.
```
%%shell

curl -o /content/feast_driver_ranking_tutorial/driver_orders.csv https://raw.githubusercontent.com/feast-dev/feast-driver-ranking-tutorial/master/driver_orders.csv 
ls /content/feast_driver_ranking_tutorial/
```

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/3.PNG)

---

### 5. parquet 과 csv 파일을 확인
```
%%shell
parq ./feast_driver_ranking_tutorial/data/driver_stats.parquet --tail 10 
```

`driver_stats.parquet file`

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/4.PNG)

`driver_orders.csv file`

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/5.PNG)


---

### 6. 데이터 로드 & 학습 (오프라인)

* 데이터를 feature_store에서 가져와 entity를 기준으로 join
* join한 데이터를 이용해 선형회귀 학습 

```
import feast
from joblib import dump
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load driver order data
orders = pd.read_csv("/content/feast_driver_ranking_tutorial/driver_orders.csv", sep="\t")
orders["event_timestamp"] = pd.to_datetime(orders["event_timestamp"])

orders.sort_values(by=['driver_id'], inplace=True, ignore_index=True)

orders['event_timestamp'][:3] = pq_[pq_['driver_id']==1001].iloc[:3].reset_index()['event_timestamp']
orders['event_timestamp'][3:6] = pq_[pq_['driver_id']==1002].iloc[:3].reset_index()['event_timestamp']
orders['event_timestamp'][6:9] = pq_[pq_['driver_id']==1003].iloc[:3].reset_index()['event_timestamp']
orders['event_timestamp'][9:10] = pq_[pq_['driver_id']==1004].iloc[:1].reset_index()['event_timestamp']

# Connect to your feature store provider
fs = feast.FeatureStore(repo_path="/content/feast_driver_ranking_tutorial")

# Retrieve training data
training_df = fs.get_historical_features(
    entity_df=orders,
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    ],
).to_df()

print("----- Feature schema -----\n")
print(training_df.info())

print()
print("----- Example features -----\n")
print(training_df.head())

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

print(training_df)

# Train model
target = "trip_completed"

print("------------------------------------------------")
reg = LinearRegression()
train_X = training_df[training_df.columns.drop(target).drop("event_timestamp")]
train_Y = training_df.loc[:, target]

reg.fit(train_X[sorted(train_X)], train_Y)

# Save model
dump(reg, "driver_model.bin")
```

`실행결과`

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/7.PNG)


`training_df`

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/8.PNG)

---

### 7. materialize into Online store

* 온라인 feature store에 데이터를 등록


```
feast materialize-incremental 2022-01-01T00:00:00
```
![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/9.PNG)

---

### 8. Predict Using Online store data

* Online store에 있는 데이터중 가장 최근 데이터 로드
* 로드한 데이터를 Predict

```
import pandas as pd
import feast
from joblib import load


class DriverRankingModel:
    def __init__(self):
        # Load model
        self.model = load("/content/driver_model.bin")

        # Set up feature store
        self.fs = feast.FeatureStore(repo_path="/content/feast_driver_ranking_tutorial/")

    def predict(self, driver_ids):
        # Read features from Feast
        driver_features = self.fs.get_online_features(
            entity_rows=[{"driver_id": driver_id} for driver_id in driver_ids],
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
                "driver_hourly_stats:event_timestamp"
            ],
        )
 
        print(driver_features.to_dict())

        print('  \n  ')

        df = pd.DataFrame.from_dict(driver_features.to_dict())
        
        print(df)

        # Make prediction
        df["prediction"] = self.model.predict(df[sorted(df)])

        print('  \n  ')

        print(df)

        print('  \n  ')


        for i in range(4):
          print(f"drvier_id : {df['driver_id'][i]}, pred_ : {df['prediction'][i]}")
        # Choose best driver
        best_driver_id = df["driver_id"].iloc[df["prediction"].argmax()]

        # return best driver
        return best_driver_id


def make_drivers_prediction():
    drivers = [1001, 1002, 1003, 1004]
    model = DriverRankingModel()
    best_driver = model.predict(drivers)
    print(f"Prediction for best driver id: {best_driver}")
```


```
make_drivers_prediction()
```

`학습한 df`

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/11.PNG)


* 이중 가장 최근 데이터를 이용

`result`

![python_exec](https://github.com/juniroc/ML_ops/blob/main/feast_/capture/10.PNG)
