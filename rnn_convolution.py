# Import libraries
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, Conv2D
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# fix random seed for reproducibility
np.random.seed(42)

# Import dataset
dataset = pd.read_csv('dummy_dataset.csv')

sentences = dataset.QUERY.values
states = dataset.STATE.values

output_dim = len(set(states))

txt = ''
# create a character set
for sentence in sentences:
    for s in sentence:
        txt += s
chars = set(txt)
input_dim = len(chars)  # this will be the dimension of the encoded letters
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# print some stats
print('number of unique states:', output_dim)
print('total chars:', input_dim)
lengths = [len(sent) for sent in sentences]
print('max, min, average number of chars per sentence:', max(lengths), min(lengths), np.average(lengths))

# create training examples and targets
# max_len = int(np.percentile(lengths, 95))  # max length of sentence (char count)
max_len = 20

X = np.ones((len(sentences), max_len, input_dim), dtype=np.int64) * -1
y = np.array(states)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence[-max_len:]):
        one_hot_encoded = np.zeros(input_dim)
        one_hot_encoded[char_indices[char]] = 1.
        X[i, (max_len - 1 - t), :] = one_hot_encoded

# Encoding the Independent Variable
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape([y.shape[0], 1])).toarray()

X_train = X[:-4, :, :]
X_test = X[-4:, :, :]
y_train = y[:-4]
y_test = y[-4:]
# X_train = np.reshape(X_train, (*X_train.shape, 1))
# X_test = np.reshape(X_test, (*X_test.shape, 1))


# Define callback for early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Initializing the RNN
classifier = Sequential()
classifier.add(Conv1D(filters=16,
                      kernel_size=3,  # should correspond to trigrams
                      strides=1,
                      activation='relu',
                      input_shape=X_train.shape[1:]))
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

print(classifier.summary())

# Fitting the RNN to the training set
history = classifier.fit(x=X_train,
                         y=y_train,
                         validation_data=(X_test, y_test),  # TODO: replace with validation!
                         epochs=300,
                         batch_size=1,
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
