from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('KNN.pkl', 'rb'))
model1 = pickle.load(open('SVM.pkl', 'rb'))
model2 = pickle.load(open('TREE.pkl', 'rb'))
model3 = pickle.load(open('NAIVE_BAYES.pkl', 'rb'))
model4 = pickle.load(open('regression.pkl', 'rb'))



app = Flask(__name__)



@app.route('/')
def man():
    return render_template('interface_home.html')
@app.route('/knn')
def knn():
    return render_template('interface_banque.html')
@app.route('/svmm')
def svmm():
    return render_template('svm.html')
@app.route('/treee')
def treee():
    return render_template('tree.html')
@app.route('/naive_bayess')
def naive_bayess():
    return render_template('naive_bayes.html')
@app.route('/regressionn')
def regressionn():
    return render_template('regression.html')
@app.route('/home', methods=['POST'])
def home(): 
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model2.predict(features)   
    return render_template('s.html', data=pred)
@app.route('/svm', methods=['POST'])
def svm(): 
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model2.predict(features)   
    return render_template('result_svm.html', data=pred)
@app.route('/tree', methods=['POST'])
def tree(): 
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model2.predict(features)   
    return render_template('result_tree.html', data=pred)
@app.route('/naive_bayes', methods=['POST'])
def naive_bayes(): 
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model2.predict(features)   
    return render_template('result_naive.html', data=pred)
@app.route('/regression', methods=['POST'])
def regression(): 
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model2.predict(features)   
    return render_template('result_regression.html', data=pred)
if __name__ == "__main__":
    app.run(debug=True)















