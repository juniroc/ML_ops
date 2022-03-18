# -*- coding: utf-8 -*-
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
