import numpy as np
import pandas as pd
import bentoml
import xgboost
from bentoml.io import NumpyNdarray, PandasDataFrame

# Load runner for the latest lightgbm model 
xgb_runner = bentoml.xgboost.load_runner("xgb_model:latest")

# Creating 'lgbm_classifier' service 
xgb_model = bentoml.Service("xgb_classifier", runners=[xgb_runner])

# Create API function and setting input format
@xgb_model.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(input_arr):
    res = xgb_runner.run_batch(input_arr)  
    return res
