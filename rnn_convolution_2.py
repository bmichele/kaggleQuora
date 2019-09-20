# Test Neural Network consisting of a bidirectional LSTM followed by a convolution and maxpooling layer. The initial RNN
# takes a sequence of one-hot-encoded characters as input
# Import libraries
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling1D, Conv1D
from keras.layers import LSTM
from keras.layers import Bidirectional
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# fix random seed for reproducibility
np.random.seed(42)

# importing dataset
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Define callback for early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

max_len = X_train.shape[1]
output_dim = 2

# Initializing the RNN
classifier = Sequential()
n_lstm = 32
classifier.add(Bidirectional(LSTM(n_lstm,
                             input_shape=X_train.shape[1:],
                             return_sequences=True)))

classifier.add(Conv1D(filters=16,
                      kernel_size=3,  # should correspond to trigrams
                      strides=1,
                      activation='relu',
                      input_shape=(max_len, n_lstm)))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Bidirectional(LSTM(32,
                                  input_shape=((max_len - 2)/2, 16))))  # ((max_len - kernel_size + 1)/pool_size, filters)
classifier.add(Dense(output_dim,
                     activation='softmax'))

# defining optimizer
opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# Compiling the RNN
classifier.compile(optimizer=opt,  # rmsprop usually good for RNN
                   # loss='mean_squared_error')
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# print(classifier.summary())

# Fitting the RNN to the training set
history = classifier.fit(x=X_train,
                         y=y_train,
                         validation_data=(X_test, y_test),  # TODO: replace with validation!
                         epochs=100,
                         batch_size=1024,
                         callbacks=[es, mc])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# load best model
saved_model = load_model('best_model.h5')

# Predict the Test set results
y_pred = saved_model.predict(X_test)
print(y_pred)
y_pred = y_pred == np.amax(y_pred, axis=1, keepdims=True)  # translate prediction in order to compare with y_test

print(y_test == 1.)
print(y_pred)
