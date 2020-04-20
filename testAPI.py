import numpy as np
from flask import Flask, jsonify, request, json
import tensorflow as tf
from numpy import array
import pandas as pd




#create my flask app Not much to say here. Set debug to False if you are deploying this API to production.
app = Flask(__name__)



# Recreate the exact same model purely from the file
new_model = tf.keras.models.load_model('testSave')

#X_test = array([[0.0,0.6,0.506944,0.6,0.055236,0.374126,0.724907,0.199052,0.786765,0.302954,0.261838,0.031014,0.35467,0.223484,0.405913,1.0,0.034749]])


# routes
@app.route('/predictMontly', methods=['POST'])

def predict():
    
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    #convert dataframe to numpy array
    arr = data_df.to_numpy()
    
    # predictions
    result = new_model.predict(arr)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)


#http://127.0.0.1:5000/predictMontly


