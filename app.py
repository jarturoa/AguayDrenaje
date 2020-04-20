import numpy as np
from flask import Flask, jsonify, request, json
import tensorflow as tf
from numpy import array, concatenate
import pandas as pd
from flask_jsonpify import jsonpify
import json
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path



#create my flask app Not much to say here. Set debug to False if you are deploying this API to production.
app = Flask(__name__)

modeloDiario = Path('/modeloDiario')
scaler = Path("testScaler")

# Recreate the exact same model purely from the file
importedMdl = tf.keras.models.load_model(modeloDiario)



scaler_filename = scaler


importedScalar = joblib.load(scaler_filename)


#print(importedScalar.data_max_)
#print(importedScalar.data_min_)

zero = array([[0]])
#X_test1 = arrOriginalay([[0.0,0.6,0.506944,0.6,0.055236,0.374126,0.724907,0.199052,0.786765,0.302954,0.261838,0.031014,0.35467,0.223484,0.405913,1.0,0.034749]])
#X_text1 = arrOriginalay([[1,4,4289948.83776,229.5,84.39582472,101.50930605,461.955377,942.071072,268.45742538,831.760486,5.17430607,1.62359422,16.2171337,49.20342242,90.52431235,29.19027778,944.87321779]])

#add a zero in a new column of arrOriginal
#db_prediction1 = np.column_stack([X_text1,zero[:,-1]])
    #normalize
#np.set_printoptions(suppress=True)
#db_finalpred = importedScalar.fit_transform(db_prediction1)
#lists2 = db_finalpred.tolist()

#print(lists2)
#prediction2 = importedMdl.predict(X_test1)

#print(prediction2)

# routes
@app.route('/predictMontly', methods=['POST'])

def predict():
    
    # get data
    data = request.get_json(force=True)


    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    
    #convert dataframe to numpy array
    arrOriginal = data_df.to_numpy()
    
    #add a zero in a new column of arrOriginal
    db_prediction = np.column_stack([arrOriginal,zero[:,-1]])
    #normalize
    np.set_printoptions(suppress=True)
    db_finalpred = importedScalar.transform(db_prediction)
    
    arrOriginal = db_finalpred[:,:-1]

    
    jsonfiles = json.loads(data_df.to_json(orient='records'))
    
    lists = arrOriginal.tolist()
    json_str = json.dumps(lists)
        
    
    # predictions
    result = importedMdl.predict(arrOriginal)  
    
    #return jsonify(json_str2)
    
    predplusnormalized = np.column_stack([arrOriginal,result[:,-1]])
    inverted = importedScalar.inverse_transform(predplusnormalized)
    result = inverted[:,17]
    
    
    #predictionToInvert = concatenate((arrOriginal[:,:-1], result), axis=1)
    #predictionToInvert = importedScalar.inverse_transform(predictionToInvert)

        
  

    # send back to browser
    output = {'results': str(result)}

    
    # return data
    return jsonify(output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)


