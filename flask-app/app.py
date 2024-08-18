from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import pickle
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/avi350751/mlops-new-mini-project.mlflow')
dagshub.init(repo_owner='avi350751', repo_name='mlops-new-mini-project', mlflow=True)

app = Flask(__name__)

#load model from model registry
model_name = 'my_model'
model_version = 5
model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)


@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['text']
    
    #clean
    text = normalize_text(text)

    #apply bow
    features = vectorizer.transform([text])

    #final prediction
    result = model.predict(features)
   
    return render_template('index.html', result = result[0])

app.run(debug=True)
