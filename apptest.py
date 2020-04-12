import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



#create my flask app Not much to say here. Set debug to False if you are deploying this API to production.
if __name__ == '__main__':
    app.run(debug=True)


#loading the pickel
model = pickle.load(open('model.pkl', 'rb'))
"""
#another way of doing it???
clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)
"""
#load the api
#Instantiate Api object
api = Api(app)

"""
The parser will look through the parameters that a user sends to your API. The parameters will be in a Python dictionary or JSON object. For this example, we will be specifically looking for a key called query. The query will be a phrase that a user will want our model to make a prediction on whether the phrase is positive or negative.

"""
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

# TODO how to import the model and run a test on it ARTURO
import tensorflow as tf

# Recreate the exact same model purely from the file
new_model = new_model = tf.keras.models.load_model('testSave')

prediction = new_model.predict(X_test)

import numpy as np


db_prediction = np.column_stack([X_test,prediction])
db_real = np.column_stack([X_test,y_test])

np.set_printoptions(suppress=True)
db_finalpred = ds_s.inverse_transform(db_prediction)
db_finalreal = ds_s.inverse_transform(db_real)

y_predicted = db_finalreal[:,17]
y_real = db_finalpred[:,17]

print(y_predicted)
print(y_real)
score = new_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test real error:', score[1])




"""
#this is your home page by defaul root page is / then it will render the template index.html
@app.route('/')
def home():
    return render_template('index.html')
"""
	
"""
i have also created the /predict wich is basically a post method I will be providing features to my model
so that my model takes input and gives us an output
@app.route('/PredictAbasto',methods=['GET'])
"""


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictAbasto, '/')

"""
Resources are the main building blocks for Flask RESTful APIs. Each class can have methods that correspond to HTTP methods such as: GET, PUT, POST, and DELETE. GET will be the primary method because our objective is to serve predictions. In the get method below, we provide directions on how to handle the user’s query and how to package the JSON object that will be returned to the user.
"""
class PredictAbasto(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
		#convert user query into list and then np.array for predict to use
		#!!!check how to use astring.split correctly, you were not signaling a variable to work with.
		user_querry = np.array(astring.split(user_query,','))
		#convert user_querry to np.array and make the prediction Returns the predicted class in an array
        prediction = model.predict(user_querry)
		
		db_finalpred = ds_s.inverse_transform(db_prediction)

        #Return the MSE of the prediction
		#pred_proba = model.predict_proba(uq_vectorized)
		
        # create JSON object
        output = {'prediction': prediction}
        
        return output


#second test for predict maybe delete later
def predict():
    '''
    For rendering results on HTML GUI from first example 
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