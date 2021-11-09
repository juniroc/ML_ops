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
![image](/uploads/556b333575d11ebf367019f791f0fcd5/image.png)


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
![image](/uploads/8a54bdbc60af582f7b8d55f3a56b37f1/image.png)

### get_statistics 
```
train_sample = 'train_df.csv'
val_sample = 'val_df.csv'
train_stats = tfdv.generate_statistics_from_csv(data_location = train_sample, delimiter=',')
val_stats = tfdv.generate_statistics_from_csv(data_location = val_sample, delimiter=',')
```

`train_stats`
![image](/uploads/0322cdfeafa2108ba17d49db36f6b6c6/image.png)

```
train_schema = tfdv.infer_schema(train_stats)
val_schema = tfdv.infer_schema(val_stats)

tfdv.display_schema(train_schema)
```

![image](/uploads/5d387637753562f619443d8ef13b0a5e/image.png)
![image](/uploads/cb4b84dce5dbfdf472220b69edce9b72/image.png)

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
![image](/uploads/1482d2c30271e3094414622ab6a0b6de/image.png)
![image](/uploads/4f07152d63b0b4d86b4c945c73d05fa3/image.png)


### Anomaly check in TFDV (highlight)
- val_stats 의 통계 결과와 train_schema 데이터와의 비교를 통해 anomaly 추출
- val_sets 에는 포함되지만 train_schema 데이터에는 포함되지 않거나 크게 튀는 데이터를 찾아냄
- 아래 사진에서 missing 데이터 `S7-P32`, `S7-P33` 이 확인
```
anomalies = tfdv.validate_statistics(statistics=val_stats, schema=train_schema)

tfdv.display_anomalies(anomalies)
```
![image](/uploads/9ea6f05df2cfba4147e084f3f89e6bce/image.png)


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
![image](/uploads/a1d7aa20571cb8881c0f952209b4424f/image.png)