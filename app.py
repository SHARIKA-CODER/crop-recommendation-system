import numpy as np
from flask import Flask , request, render_template
import pickle
from werkzeug.wrappers import Request,Response
import pandas as pd


app =Flask(__name__,template_folder='template')
model1 = pickle.load(open("model.pkl","rb"))
model2 = pickle.load(open("model1.pkl","rb"))
model3 = pickle.load(open("model2.pkl","rb"))
model4 = pickle.load(open("model3.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/Bdm")
def Bdm():
    return render_template("bagging decision model.html")
@app.route("/predictBdm",methods=["POST"])
def predictBdm():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model1.predict(features)
    label_name= pd.read_csv('label_name_number.csv')
    result=list(label_name[label_name['crop_number']==int(prediction)]['crop_name'])
    value = result[0]
    return render_template("bagging decision model.html", prediction_text="The suitable crop is {}".format(value))


@app.route("/dm")
def dm():
    return render_template("decision model.html")  
@app.route("/predictdm", methods =["POST"])
def predictdm():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model2.predict(features)
    label_name= pd.read_csv('label_name_number.csv')
    result=list(label_name[label_name['crop_number']==int(prediction)]['crop_name'])
    value = result[0]
    return render_template("decision model.html", prediction_text="The suitable crop is {}".format(value))

@app.route("/NM")
def NM():
    print("Rotued")
    return render_template("neuralnetwork.html")
@app.route("/predictnm",methods=['POST'])
def predictNM():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model3.predict(features)
    label_name= pd.read_csv('label_name_number.csv')
    result=list(label_name[label_name['crop_number']==int(prediction)]['crop_name'])
    value = result[0]
    return render_template("neuralnetwork.html", prediction_text="The suitable crop is {}".format(value))

@app.route("/bnm")
def bnm():
    return render_template("bagging neural network.html")

@app.route("/predictbnm", methods =["POST"])
def predictbnm():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model3.predict(features)
    label_name= pd.read_csv('label_name_number.csv')
    result=list(label_name[label_name['crop_number']==int(prediction)]['crop_name'])
    value = result[0]
    return render_template("neuralnetwork.html", prediction_text="The suitable crop is {}".format(value)) 
      
  
@app.route("/about")
def about():
    return render_template("about.html")


if __name__== "__main__":
    app.run(debug=True)
