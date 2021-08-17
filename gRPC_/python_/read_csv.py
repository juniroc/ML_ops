import pandas as pd
df = pd.read_csv('./datas/todo_lst.csv', index_col='name')
print(df)
# for i in range(len(df)):
#     print(df.iloc[i,0])
dict_ = df.to_dict(orient='dict')
dict_ = dict_['value']
print(dict_)