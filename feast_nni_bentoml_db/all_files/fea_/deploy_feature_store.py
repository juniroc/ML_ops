# This is an example feature definition file

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
    entities=["ticket_id"],
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
