import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

def get_type(text):
    if text=="Intel Core i3" or text=="Intel Core i5" or text=="Intel Core i7":
        return text
    elif "Intel" in text:
        return "Other Intel Processor"
    else:
        return "AMD processor"

def OS_catagory(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

def predict_price(My_input):
    df=pd.read_csv('./laptop_data.csv')
    df.drop(['Unnamed: 0','Weight'],axis=1 ,inplace=True)
    df['Touchscreen']=df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['IPS']=df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    df['Y_res']=df['ScreenResolution'].apply(lambda x: x.split('x')[1])
    df['Y_res']=df['Y_res'].astype('int32')
    df['X_res']=df['ScreenResolution'].apply(lambda x: x.split('x')[0]).astype('string')
    df['X_res']=df['X_res'].apply(lambda x: x[-4:]).astype('int32')
    df['PPI']=(((df['X_res']**2)+(df['Y_res']**2))**0.5)/df['Inches']
    df.drop(['ScreenResolution','Inches','X_res','Y_res'],axis=1,inplace=True)
    df['Ram']=df['Ram'].apply(lambda x: x.split('GB')[0])
    df['Ram']=df['Ram'].astype('int32')

    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')
    new = df["Memory"].str.split("+", n = 1, expand = True)
    df["first"]= new[0]
    df["first"]=df["first"].str.strip()
    df["second"]= new[1]
    df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
    df['first'] = df['first'].str.replace(r'\D', '')
    df["second"].fillna("0", inplace = True)
    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)
    df['second'] = df['second'].str.replace(r'\D', '')
    df["first"] = df["first"].astype(int)
    df["second"] = df["second"].astype(int)
    df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
    df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
    df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
    df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])
    df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'],inplace=True)

    df=df.drop(['Memory'],axis=1)
    df['CPU Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
    df=df.drop(['Cpu'],axis=1)
    df['CPU Name']=df['CPU Name'].apply(get_type)
    df['Gpu']=df['Gpu'].str.split(' ').apply(lambda x: x[0])
    df=df[df['Gpu']!='ARM']
    df.rename(columns={"Gpu":"GPU Name"},inplace=True)
    df['OS'] = df['OpSys'].apply(OS_catagory)
    df=df.drop(['OpSys'],axis=1)
    Y=df['Price']
    X=df.drop(['Price'],axis=1)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=2)
    step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,11,12])],remainder='passthrough')
    step2 = RandomForestRegressor(n_estimators=100,random_state=3,max_features=0.75,max_depth=15)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(X_train,Y_train)
    Y_pred = pipe.predict(My_input)
    print("Predicted price=",Y_pred)
    return Y_pred

if __name__=="__main__":
    predict_price()