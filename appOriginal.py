import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create my flask app
app = Flask(__name__)
#loading the pickel
model = pickle.load(open('model.pkl', 'rb'))

#this is your home page by defaul root page is / then it will render the template index.html
@app.route('/')
def home():
    return render_template('index.html')
"""
i have also created the /predict wich is basically a post method I will be providing features to my model
so that my model takes input and gives us an output
"""

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #input from all the forms
    #request allows you to store all the values from the textfield and store them in int_features
    #change to double so that it can accept decimals
    #int_features = [int(x) for x in request.form.values()]
    int_features = [float(x) for x in request.form.values()]
    #convert into an array
    final_features = [np.array(int_features)]
    
    prediction = model.predict(final_features)
    	
    #I am finally getting the output
    output = round(prediction[0], 2)
    	
    	
    return output	
    #return render_template('index.html', prediction_text='Yo popcorn ready, Qsuministrada is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)