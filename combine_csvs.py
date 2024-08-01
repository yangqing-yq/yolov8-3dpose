import csv

## valid subdirs
subdirs=['Interview','SignLanguage','TalkShow','Entertainment','LiveVlog',\
            'Olympic','TVShow','Fitness','Magic_show','Online_class','VideoConference']



import pandas as pd 
  
# reading two csv files 

df=[]
for i,subdir in enumerate(subdirs):
     if i==0:
          df.append(pd.read_csv('../eval_csvs/'+subdir+'_eval.csv'))
     else:
          print("i:",i)
          df.append(pd.read_csv('../eval_csvs/'+subdir+'_eval.csv'))
          print(df[i][subdir])
          df[i]=pd.concat([df[i-1],df[i][subdir]],axis=1)

df[len(df)-1].to_csv("../eval_csvs/res_ubody.csv",index=False)   


