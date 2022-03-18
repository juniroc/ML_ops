import requests
import pandas as pd
import socket
df= {'time' : [2.78333, 3.183333, 4.4167775, 5.483313, 15.522422, 18.652444], 
     'weekday' : [2, 3, 4, 5, 5, 1], 
     'weekend' : [0, 1, 1, 0, 1, 1], 
     'instlo_1': [3, 12, 3, 11, 11, 11],
     'instlo_2': [128, 55, 37, 34, 45, 23],
     'inst_code' : [142, 1215, 133, 44, 13, 23], 
     'sysname_lo': [1515, 552, 2100, 113, 133, 799], 
     'sysname_eq' : [0, 0, 0, 0, 0, 1]}
df_ = pd.DataFrame(df)

print(df_.to_json())

print(df_)
response = requests.post(
#    "http://127.0.0.1:3000/predict",
    "http://210.114.89.130:30050/predict",
    headers={"content-type": "application/json"},
    data=df_.to_json())

print(response.text)

print(socket.gethostbyname(socket.gethostname()))

