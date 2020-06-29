# load libraries
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras import backend
import matplotlib.pyplot as plt

# define RMSE metric
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# set working directory
os.chdir(r'C:\Users\Peter A Hall\Documents\GitHub\neural_network_stock_trading')

# read in training data and split
trainingData = pd.read_csv("trainingData.csv", header = 0)
xTrain = trainingData.iloc[ : , 0:41]
yTrain = trainingData.iloc[ : , 41]
yTrain.name = 'y_Actual'  # rename yTrain series

# ANN define the model
model = Sequential()
model.add(Dense(units = 100, activation='relu', input_dim = 41))  #hidden layer 1 with input
model.add(Dropout(0.2))
model.add(Dense(units = 100, activation='relu'))
model.add(Dense(units = 80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 80, activation='relu'))
model.add(Dense(units = 15, activation='relu'))
model.add(Dense(units=1, activation = 'sigmoid'))   #output layer

# compile the model
model.compile(
        loss = 'binary_crossentropy', 
        optimizer = 'adam', 
        metrics = ['accuracy', rmse]
)

# fit the model
history = model.fit(xTrain, yTrain, epochs=10, verbose=1)

# test the model, convert np.array to pd.Series, and print confusion matrix
y_PredictedArray = model.predict_classes(xTrain)
y_Predicted = pd.Series(y_PredictedArray.flatten('C'), name = 'y_Predicted')
y_Actual = yTrain
resultsDF = pd.concat([y_Actual, y_Predicted], axis=1)
confusion_matrix = pd.crosstab(resultsDF['y_Actual'], resultsDF['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

# plot of accuracy and RMSE across epochs
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.ylabel('Accuracy')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False
) # labels along the bottom edge are off
plt.subplot(2, 1, 2)
plt.plot(history.history['rmse'])
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()

