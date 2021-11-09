### library 
```
import os
import tfx
import pandas as pd
import numpy as np
import tensorflow as tf
from tfx.components import CsvExampleGen
import tensorflow as tf
import tensorflow_data_validation as tfdv

# from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
```

### TFrecord
```
with tf.io.TFRecordWriter("test.tfrecord") as w:
    w.write(b"First record")
    w.write(b"Second record")

for record in tf.data.TFRecordDataset("test.tfrecord"):
    print(record)
```
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/1.png)


### dataset for TFDV
```
csv_path = "SAMPLE_ALARM(7_9월_라벨완).csv"
df = pd.read_csv(csv_path, encoding='cp949')
df = df[df['ticket_id'].notna()].reset_index()

train_df = df.iloc[:-500, :]
val_df = df.iloc[-500:, :]

train_df.to_csv('./train_df.csv')
val_df.to_csv('./val_df.csv')

df

```
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/2.png)

### get_statistics 
```
train_sample = 'train_df.csv'
val_sample = 'val_df.csv'
train_stats = tfdv.generate_statistics_from_csv(data_location = train_sample, delimiter=',')
val_stats = tfdv.generate_statistics_from_csv(data_location = val_sample, delimiter=',')
```

`train_stats`
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/3.png)

```
train_schema = tfdv.infer_schema(train_stats)
val_schema = tfdv.infer_schema(val_stats)

tfdv.display_schema(train_schema)
```

![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/4.png)
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/5.png)

### visualize_statistics
```
train_stats = tfdv.generate_statistics_from_csv(
data_location=train_sample)
val_stats = tfdv.generate_statistics_from_csv(
data_location=val_sample)

tfdv.visualize_statistics(lhs_statistics=val_stats,
rhs_statistics=train_stats,
lhs_name='VAL_DATASET', rhs_name='TRAIN_DATASET')
```
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/6.png)
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/7.png)


### Anomaly check in TFDV (highlight)
- val_stats 의 통계 결과와 train_schema 데이터와의 비교를 통해 anomaly 추출
- val_sets 에는 포함되지만 train_schema 데이터에는 포함되지 않거나 크게 튀는 데이터를 찾아냄
- 아래 사진에서 missing 데이터 `S7-P32`, `S7-P33` 이 확인
```
anomalies = tfdv.validate_statistics(statistics=val_stats, schema=train_schema)

tfdv.display_anomalies(anomalies)
```
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/8.png)


### update value in column (train_schema)
- val_set에는 있지만 train_schema 에 존재하지 않은 경우
  - train_schema에 value(여기선 `S7-P32`, `S7-P33`)를 추가해주면 anomaly 문제 해결 가능
```
root_cause_portz_feature = tfdv.get_domain(train_schema, 'root_cause_portz')

root_cause_portz_feature.value.append('S7-P32')
root_cause_portz_feature.value.append('S7-P33')

updated_anomalies = tfdv.validate_statistics(val_stats, train_schema)

tfdv.display_anomalies(updated_anomalies)
```
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/9.png)


### Drift & Skew Check
- 각 컬럼 별로 Skew 를 확인
- 일정 Threshold를 주어 통계값(분산 차이 L-infinity)이 threshold 보다 높을 경우 찾아냄
```
# skew check each column
tfdv.get_feature(train_schema, 'ticket_fm_nonfm').skew_comparator.infinity_norm.threshold = 0.000001

skew_anomalies = tfdv.validate_statistics(statistics=train_stats, schema=train_schema, serving_statistics=val_stats)

tfdv.display_anomalies(skew_anomalies)
```
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/10.png)


- 컬럼 별로 drift 확인
- 일정 threshold를 주어 통계값(분산 차이 L-infinity)이 threshold 보다 높을 경우 찾아냄
```
# Drift check each column
tfdv.get_feature(train_schema, 'root_cause_sysnamea').drift_comparator.infinity_norm.threshold = 0.01

drift_anomalies = tfdv.validate_statistics(statistics=train_stats, schema = train_schema, previous_statistics= val_stats)

tfdv.display_anomalies(drift_anomalies)
```
![image](https://github.com/juniroc/ML_ops/blob/main/TFX/images/11.png)
