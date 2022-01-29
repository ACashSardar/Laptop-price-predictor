from flask import Flask, render_template, request
from Laptop_price_predictor import predict_price
import pandas as pd

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def first_page():
    return render_template("Homepage.html")

@app.route("/next_page",methods=['GET','POST'])
def next_page():
    Input_Dict={'Company': None, 'TypeName': None, 'Ram':None, 'GPU Name': None, 'Touchscreen': None, 'IPS':None, 
'PPI':None,'HDD': None, 'SSD': None, 'Hybrid':None, 'Flash_Storage':None, 'CPU Name': None, 'OS': None}
    try:
        if request.method=='POST':
            Input_Dict['Company']=request.form.get('Brand_name')
            Input_Dict['TypeName']=request.form.get('Type_name')
            Input_Dict['Ram']=request.form.get('RAM_size')
            Input_Dict['GPU Name']=request.form.get('GPU_brand')
            Input_Dict['OS']=request.form.get('Op_sys')
            Input_Dict['Touchscreen']=request.form.get('Touchscreen')
            Input_Dict['IPS']=request.form.get('IPS')
            Input_Dict['PPI']=request.form.get('PPI')
            Input_Dict['HDD']=request.form.get('HDD')
            Input_Dict['SSD']=request.form.get('SSD')
            Input_Dict['Hybrid']=request.form.get('Hybrid')
            Input_Dict['Flash_Storage']=request.form.get('Flash_Storage')
            Input_Dict['CPU Name']=request.form.get('CPU_brand')
            df = pd.DataFrame.from_records([Input_Dict])
            df['Ram']=df['Ram'].astype('int32')
            df['Touchscreen']=df['Touchscreen'].apply(lambda x: 1 if x=='Yes' else 0)
            df['IPS']=df['IPS'].apply(lambda x: 1 if x=='Yes' else 0)
            df['PPI']=df['PPI'].astype('float64')
            df['HDD']=df['HDD'].astype('int64')
            df['SSD']=df['SSD'].astype('int64')
            df['Hybrid']=df['Hybrid'].astype('int64')
            df['Flash_Storage']=df['Flash_Storage'].astype('int64')
            print(df,df.info())
        return render_template("Homepage.html",price=int(predict_price(df)[0]))
    except:
        return render_template("warning.html")

if __name__=="__main__":
    app.run(debug=True,port=5000)