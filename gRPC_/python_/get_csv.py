import pandas as pd
a_dictionary = {}
df = pd.DataFrame.from_dict(a_dictionary, orient='index',columns=['value'])
print(df)

df.to_csv('./datas/todo_lst.csv',index_label='name')