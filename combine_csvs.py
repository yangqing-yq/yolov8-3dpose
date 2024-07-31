import csv

## valid subdirs
subdirs=['Interview','SignLanguage','TalkShow','Entertainment','LiveVlog',\
            'Olympic','TVShow','Fitness','Magic_show','Online_class','VideoConference']



import pandas as pd 
  
# reading two csv files 

dataall = pd.read_csv('../eval_csvs/'+'all_eval.csv') 
df=[]
for i,subdir in enumerate(subdirs):
     df.append(pd.read_csv('../eval_csvs/'+subdir+'_eval.csv'))
     print(df[i][subdir])
     dataall=pd.concat([dataall,df[i][subdir]],axis=1)

dataall.to_csv("../eval_csvs/res_ubody.csv",index=False)   

    #  dataall = pd.merge(dataall, '../eval_csvs/'+subdir+'_eval.csv',  
    #                on='LOAN_NO',  
    #                how='inner') 

  
# using merge function by setting how='inner' 

  
# displaying result 
# print(dataall) 

