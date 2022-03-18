# -*- coding: utf-8 -*-

import torch
import lightgbm as lgb
import pandas as pd
import bentoml
import xgboost

# get_model
mo_ = torch.load('./models/lgbm_model.pth.tar')['model']

# dr_lauren_classifier와 'model'로 패키징됨
bentoml.lightgbm.save('lgbm_model', mo_)

