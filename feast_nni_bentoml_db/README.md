---
# feast_nni_bentoML_db

## feast_

사진[4]


### 1. install feast_ and init
* fea_ 라는 이름으로 초기 폴더를 생성

```
feast init fea_
```

* 구성 파일
1. data folder
2. feature_store.yaml
3. example.py (여기서는 `deploy_feature_store.py` 라는 이름으로 수정)

사진[0]




### 2. save_parquet_data(dr_lauren)
* 전처리가 완료된 데이터를 parquet 양식으로 저장 \
(현재는 parquet format만 지원)

file_name : `ppr_data_.parquet` \
dir_path : `fea_/data/`

사진 [1]


### 3. edit `deploy_feature_store.py`
* deploy 하기 위한 python 파일 수정
* 원하는 파일 path 지정 및 event_timestamp_column 을 지정 \
(이는 추후 online_store에서 실시간으로 업로드되는 데이터 load 시 활용)

```
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Read data from parquet files. Parquet is convenient for local development mode.
dr_lauren_stat = FileSource(
    path="/workspace/ML_Ops/feast/fea_/data/ppr_data_.parquet",
    event_timestamp_column="event_timestamp",
)

# Define an entity for the driver. You can think of entity as a primary key used to
# fetch features.
driver = Entity(name="ticket_id", value_type=ValueType.INT64, description="ticket_id",)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
dr_lauren_stat_view = FeatureView(
    name="dr_lauren_stat",
    entities=["ticket_id"],     ### ticket_id 를 key 로써 entity를 지정
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="time", dtype=ValueType.FLOAT),
        Feature(name="weekday", dtype=ValueType.INT64),
        Feature(name="weekend", dtype=ValueType.INT64),
        Feature(name="instlo_1", dtype=ValueType.INT64),
        Feature(name="instlo_2", dtype=ValueType.INT64),
        Feature(name="inst_code", dtype=ValueType.INT64),
        Feature(name="sysname_lo", dtype=ValueType.INT64),
        Feature(name="sysname_eq", dtype=ValueType.INT64),
        Feature(name="ntt_label", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=dr_lauren_stat,
    tags={},
)

```


### 4. edit `feature_store.yaml`

* feature_store_view 를 저장하기 위한 `registry.db`
* 이후 materialize 명령어를 통해 `online_store.db` 로 load한다.
* local환경이므로 `provider == local`

```
project: fea_
registry: data/registry.db
provider: local
online_store:
    path: data/online_store.db
```


### 5. deploying_feature_store
```
feast apply
```

사진[2]

`data.folder`

사진[3]


### 6. get_train_from_offline_
* get_historical_features() 로 학습시킬 데이터 로드

사진[5]

* criterion 을 통해 7월 이전 데이터만 학습

```
def get_train_from_offline_store(path_):
    # Connect to your feature store provider
    fs = feast.FeatureStore(repo_path=f"{path_}")


    ### entity_df 
    parquet_ = pd.read_parquet(f'{path_}/data/ppr_data_.parquet', engine='pyarrow')
    orders = parquet_[['ticket_id','event_timestamp']]

    # Retrieve training data
    training_df = fs.get_historical_features(
        entity_df=orders,
        features=[
            "dr_lauren_stat:time",
            "dr_lauren_stat:weekday",
            "dr_lauren_stat:weekend",
            "dr_lauren_stat:instlo_1",
            "dr_lauren_stat:instlo_2",
            "dr_lauren_stat:inst_code",
            "dr_lauren_stat:sysname_lo",
            "dr_lauren_stat:sysname_eq",
            "dr_lauren_stat:ntt_label",
        ],
    ).to_df()


    ### training_part_before 7 month

    criterion = '2021-07-01'

    training_df_ = training_df[training_df['event_timestamp'] < criterion]
    
    return training_df_


get_train_from_offline_store('/workspace/ML_Ops/feast/fea_')

```

사진[6]





## nni_from_feast

### 1. nni config
* hyperparameter 를 위한 search_space 생성 \
(우선적으로 lgb, xgb 만 적용)
1. model_name
2. n_estimators (boosting tree 갯수)
3. seed 

`search_space.json`
```
{
  "model_n": {
    "_type": "choice",
    "_value": ["lgb", "xgb"]
  },
  "ne": {
    "_type": "choice",
    "_value": [50, 60, 70, 80, 90, 100]
  },
  "seed": {
    "_type": "choice",
    "_value": [42, 32, 16, 8]
  }
}
```

* nni 학습 python file 생성

* acc를 우선적으로 평가하는 nni 생성
* 이후 생성된 모델 이름은 `model_acc_auc_` 로 저장

`test_.py`
```
parser = argparse.ArgumentParser(description='sklearn Training')

parser.add_argument('--model-dir', default='./models_', type=str)
parser.add_argument('--seed', default=42, type=int,
                    help='the number of seed')
parser.add_argument('--request-from-nni', default=False, action='store_true')
parser.add_argument('--model-n', default="lgb", type=str,
                    help='initial learning rate')

parser.add_argument('--ne', '--n-estimators', default=100, type=int,
                    help='initial learning rate')

parser.add_argument('--gpu', default=None, type=int,
                    help='initial learning rate')
```

```
def main():
    args = parser.parse_args()
    
    if args.request_from_nni:
        import nni
        tuner_params = nni.get_next_parameter()
#         logger.info(str(tuner_params))

            
        if "ne" in tuner_params:
            args.ne_ = tuner_params["ne"]

        if "model_n" in tuner_params:
            args.model_ = tuner_params["model_n"]
            
        if "seed" in tuner_params:
            args.seed_ = tuner_params["seed"]


#         logger.info(str(args))

        
        # demonstrate that intermediate result is actually sent
        nni.report_intermediate_result(0.)

        args.model_dir = '/workspace/ML_Ops/feast/models_'



    if args.gpu is not None:
        args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    path_ = '/workspace/ML_Ops/feast/fea_/'
    #### from feature offline store
    train_df = get_train_from_offline_store(path_)
    
    args.train_df_ = train_df

    # Simply call main_worker function
    main_worker(args.gpu, args)
```

```
def main_worker(gpu, args):
    best_acc = 0
    best_auc = 0

    device = args.gpu

    df_ = args.train_df_


    if args.model_ == "lgb":

        mo_, acc_, f1_score_, auc_ = get_lgbm_score(args.train_df_, 0.3, args.ne_, args.seed_)

    elif args.model_ == "xgb":

        mo_, acc_, f1_score_, auc_ = get_xgb_score(args.train_df_, 0.3, args.ne_, args.seed_)



    # remember best acc@1 and save checkpoint

    acc_best = acc_ > best_acc
    best_acc = max(acc_, best_acc)

    auc_best = auc_ > best_auc
    best_auc = max(auc_, best_auc)

    save_checkpoint({
        'acc': acc_,
        'auc': auc_,
        'seed' : args.seed_,
        'model': mo_,
    }, args.model_, acc_, auc_, args.model_dir)


    try:
        if args.request_from_nni:
            import nni
            nni.report_final_result(acc_)
    except NameError:

        pass
```
```
if __name__ == '__main__':
    main()
```

* yaml 파일을 통해 search_space 를 이용한 test_.py 실행 \
(이때 nni 적용)

`210819_.yaml`
```
authorName: nujnim
experimentName: gbm
trialConcurrency: 2
maxExecDuration: 99999d
maxTrialNum: 100
trainingServicePlatform: local
searchSpacePath: ./search_space.json
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  codeDir: .
  command: python test_.py --gpu 0 --request-from-nni 
  gpuNum: 1

```


### 2. nnictl 명령어를 이용해 실행

* 8085 port == webui port
* nnictl create 명령어를 통해 yaml 파일을 실행한다.

```
nnictl create -p 8085 --config ./210819_.yaml
```

사진[10]



### 3. check nni by web ui

사진[9]

사진[8]


### 4. check saved models_

사진[11]




## Inference_by_Online_store_with_Bentoml

### 1. create_package_python (bentoML)

* 저장된 model 을 로드 및 bentoml로 패킹


사진[12]

`model_packing_save.py`
```
import torch
import pandas as pd
import pack_ 

# create dr_lauren_service instance
dr_lauren_service = pack_.Dr_lauren_classifier()

# get_model
mo_ = torch.load('/workspace/ML_Ops/feast/fea_/models_/xgb_acc_0.97829_auc_0.99705_.pth.tar')['model']

# dr_lauren_classifier와 'model'로 패키징됨
dr_lauren_service.pack('model', mo_)

# 경로 저장
saved_path = dr_lauren_service.save()
```

`pack_.py`
```
import pandas as pd
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class Dr_lauren_classifier(BentoService):
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


### 2. packing_model

* python 명령어를 이용해 실행


```
python model_packing_save.py
```


### 3. Chcek package

* saved_path 에 들어가면 package_file 로 묶여있는 것을 확인
* path = `/root/bentoml/repository/Dr_lauren_classifier/~`


사진[13]


### 4. model serving


* 컨테이너로 띄울 수 있지만, 도커 이미지 내에서 진행하므로 바로 serving

``` 
bentoml serving Dr_lauren_classifier:latest
```

사진[14]


### 5. materialize data

* 데이터를 online_store로 load


사진[15]


```
feast materialize 2021-01-01T00:00:00 2021-08-1T00:00:00
```

사진[16]


### 6. get_predict_from_online_store

* `online_inf.py` : online_store 에서 데이터를 끌어와 inference

* inference 하고 싶은 entity_rows를 뽑는다 \
(여기서는 7월 이후 데이터)

* online_store로 부터 해당 entity에 해당하는 data를 끌어온다.


`online_inf.py`
```
def get_entities(path_,from_time_):
    parquet_ = pd.read_parquet(path_, engine='pyarrow')
    orders = parquet_[['ticket_id','event_timestamp']]

    ### 뽑을 rows의 entity == key 라고 볼 수 있음
    new_orders = orders[orders['event_timestamp']>= from_time_]
    new_orders.drop_duplicates(ignore_index=True, inplace=True)
    return new_orders
```

사진[17]

```
def get_df_from_online(feast_path_, entity_df_):
    fs_ = feast.FeatureStore(repo_path=feast_path_)

    online_ = fs_.get_online_features(
                entity_rows=[{"ticket_id": i} for i in entity_df_['ticket_id']],
                features=[
                    "dr_lauren_stat:time",
                    "dr_lauren_stat:weekday",
                    "dr_lauren_stat:weekend",
                    "dr_lauren_stat:instlo_1",
                    "dr_lauren_stat:instlo_2",
                    "dr_lauren_stat:inst_code",
                    "dr_lauren_stat:sysname_lo",
                    "dr_lauren_stat:sysname_eq",
                    "dr_lauren_stat:ntt_label",
            ],
    )


    df = pd.DataFrame.from_dict(online_.to_dict())

    return df[['time', 'weekday', 'weekend', 'instlo_1', 'instlo_2', 'inst_code', 'sysname_lo', 'sysname_eq','ntt_label']]
```

사진[18]


```
def get_from_bentoml(online_df_):
    response = requests.post("http://127.0.0.1:5000/predict", data=online_df_.iloc[:,:-1].to_json())
    infer_ = response.text
    predict_ = infer_[1:-1].split(', ')
    return predict_

def get_infer_df_(entity_df_, online_df_):
    infer_df = entity_df_.reset_index(drop=True)
    infer_df['label'] = online_df_['ntt_label']
    infer_df['predict'] = get_from_bentoml(online_df_)
    return infer_df
```

사진[19]


## to_db(mysql)


### inference_to_db (requset_to_mysql)

* online_inf 내에서 db로 연동하는 코드까지 구현

사진[20]

* entity, event_timestamp, 기존 label, inference 한 결과를 입력


`online_inf.py`
```
### data create, insert, update
def get_commit_sql(conn_, sql_):
    with conn_.cursor() as cursor:
        cursor.execute(sql_)
        result = cursor.fetchall()
        conn.commit()

### data read
def get_data_sql(conn_, sql_):
    with conn_.cursor() as cursor:
        cursor.execute(sql_)
        result = cursor.fetchall()
        return result
        
        
def get_columns_(df_):
    columns_str = ', '.join([i for i in infer_df.columns])
    return columns_str
    

def get_values_(df_):
    value_ = []
    for row_ in df_.values:
        value_.append("("+", ".join(["'"+str(i)+"'" for i in row_]) + ")")

    values_str = ', '.join(value_)
    return values_str
    
    

def pred_into_db(conn_, infer_df_):
    columns_ = get_columns_(infer_df_)

    values_ = get_values_(infer_df_)

    sql = f"INSERT INTO infer_df ({columns_}) VALUES {values_} ;"
    get_commit_sql(conn_, sql)
    
    
def main():
    path = 'workspace/ML_Ops/feast_fea_/data/ppr_data_.parquet'
    from_time = '2021-07-01'

    entity_df = get_entities(path, from_time)
    
    feast_path = './'
    online_df = get_df_from_online(feast_path, entity_df)
    
    infer_df = get_infer_df_(entity_df, online_df)
    
    conn = pymysql.connect(host='localhost', port=3306, user='root', bind_address='127.0.0.1', passwd='', db='infer_db', charset='utf8')
    
    pred_into_db(conn,infer_df)
```

사진[21]

사진[22]









---