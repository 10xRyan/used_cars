from flask import Flask,request,render_template

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('basic.html')

@app.route('/predict',methods=['GET','POST'])
def prediction():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            model=request.form.get('model'),
            year=request.form.get('year'),
            transmission=request.form.get('transmission'),
            mileage=request.form.get('mileage'),
            fuelType=request.form.get('fuelType'),
            tax=request.form.get('tax'),
            mpg=request.form.get('mpg'),
            engineSize=request.form.get('engineSize')
        )
        pred_df=data.get_data_as_df()
        print(pred_df)

        predict_pip=PredictPipeline()
        results=predict_pip.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5050,debug=True)
