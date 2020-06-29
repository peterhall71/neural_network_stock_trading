# Notes

## Pipeline

**Data Fetching**
Raw data is current stored in a network MySQL Server.
Data fetching is performed in the file designated 1.

**Preprocessing**
Data preprocessing is performed in R.
The data is reformated and recordeds are coded 1 or 0 to structure the training data.
Preprocessing occurs according the file designated 2.

**Neural Network Training**
The neural network model is built and trained using Python according to the file designated 3.
This is done using the Keras package.

## Kares

Keras input explanation: input_shape, units, batch_size, dim, etc
https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc

Choosing Optimizer
https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/

## Convolutional Neural Network

https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/