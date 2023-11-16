import pandas as pd
import numpy as np

df=pd.read_csv('./train_data.csv',index_col=0)
df.drop(columns=['day','vix','turbulence'],inplace=True)
df.fillna(0,inplace=True)
df['return_today']=(df['close']-df['open'])/df['close']

def bias(x):
    x['bias']=(x.close-x['close'].rolling(window=5).mean())/x['close'].rolling(window=5).mean()
    return x

df=df.groupby('tic').apply(lambda x: bias(x))
df.reset_index(drop=True, inplace=True)

def vroc(x):
    x['vroc']=(x['volume']-x['volume'].shift(5))/x['volume'].shift(5)
    return x

df=df.groupby('tic').apply(lambda x: vroc(x))
df.reset_index(drop=True, inplace=True)

def wr(x):
    x['wr']=(x['high'].rolling(5).max()-x['close'])/(x['high'].rolling(5).max()-x['low'].rolling(5).min())
    return x

df=df.groupby('tic').apply(lambda x: wr(x))
df.reset_index(drop=True, inplace=True)

def return_yesterday(x):
    x['return_yesterday']=(x['close']-x['close'].shift(1))/x['close'].shift(1)
    return x

df=df.groupby('tic').apply(lambda x: return_yesterday(x))
df.reset_index(drop=True, inplace=True)

def z_score(dt):
    dt1=dt.iloc[:,:2]
    dtclose=dt.iloc[:,5]
    dt2=dt.iloc[:,2:]
    mean=dt2.mean()
    std=dt2.std()
    dt2=(dt2-mean)/std
    dt2.rename(columns={'close':'close_z'},inplace=True)
    dt=pd.concat([dt1,dtclose,dt2],axis=1)
    return dt

df=df.groupby('date').apply(lambda x: z_score(x))
df.reset_index(drop=True, inplace=True)
df=df.sort_values(['date','tic'])
df.fillna(0,inplace=True)

df.to_csv('./train_data_update.csv')

df=df.drop(columns=['tic'])
date=df.date.unique()
a=np.zeros(shape=[len(date),29,18])
for i in range(len(date)):
    a[i]=df[df['date']==date[i]].drop(columns=['date']).values


np.save('./stock_features.npy',a)