import numpy as np
import pandas as pd
import bentoml
import xgboost
import lightgbm
from bentoml.io import NumpyNdarray, PandasDataFrame

# Load runner for the latest lightgbm model 
lgbm_runner = bentoml.lightgbm.load_runner("lgbm_model:latest")

# Creating 'lgbm_classifier' service 
lgbm_model = bentoml.Service("lgbm_classifier", runners=[lgbm_runner])

# Create API function and setting input format
@lgbm_model.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(input_arr):
    res = lgbm_runner.run_batch(input_arr)  
    return res
