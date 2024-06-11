# Importing essential libraries
from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart.pkl'
model = joblib.load('heart.pkl')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('basic2.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model.predict(data)
        
        return render_template('basic3.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)
