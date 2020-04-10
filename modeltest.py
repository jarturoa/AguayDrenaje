# Importing the libraries
from numpy import loadtxt
from numpy import concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

#import keras



from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.keras.models import load_model

from math import sqrt
from matplotlib import pyplot
from datetime import datetime
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pickle



# load the dataset
ds = loadtxt('C:/Users/Arturo A/AguayDrenaje/MM8.csv', delimiter=',')
#normalize
ds_s = MinMaxScaler(feature_range=(0,1))
ds = ds_s.fit_transform(ds)
# put features and prediction in separate parts
# split into input (X) and output (y) variables
X = ds[:,0:17]
y = ds[:,17]
#divide dataset train validation test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# define the keras model
#from keras.layers import LSTM
#stadard
model = Sequential()
#input 8 first hidden layer of 12
model.add(Dense(12, input_dim=17, activation='relu'))
#goes second hidden layer with 8 
model.add(Dense(24, activation='relu'))
#final output layer
model.add(Dense(1, activation='relu'))
# compile the keras model
#just a test to change the learning rate
#we tried with SDG, it didn't work
model.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mse', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])

# fit the keras model on the dataset
#use first one for final model for production
#model.fit(X_train, y_train, epochs=2000, batch_size=12)
model.fit(X_train, y_train, epochs=150, batch_size=12)

score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


"""
from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

"""


"""
USING YAML FILE IS NOT WORKING
ValueError: Unknown initializer: GlorotUniform

from keras.models import model_from_yaml

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

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
"""


#PICKLE IS NOT WORKING FOR THE NUMBER OF CELLS IN THE DNN
# Saving model to disk, you input what object how you want it named and write = wb
#pickle.dump(model, open('modeltest.pkl','wb'))

# Save the Modle to file in the current working directory

Pkl_Filename = "model2.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file, fix_imports=True)

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
print(model.predict(X_test[0, :]))


"""
# Save RL_Model to file in the current working directory

from sklearn.externals import joblib

joblib_file = "model2.pkl"  
joblib.dump(model, joblib_file)

joblib_model2 = joblib.load(joblib_file)

joblib_model2

"""

"""
dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
"""
