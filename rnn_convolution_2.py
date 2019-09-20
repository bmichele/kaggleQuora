# Test Neural Network consisting of a bidirectional LSTM followed by a convolution and maxpooling layer. The initial RNN
# takes a sequence of one-hot-encoded characters as input
# Import libraries
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling1D, Conv1D
from keras.layers import LSTM
from keras.layers import Bidirectional
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# fix random seed for reproducibility
np.random.seed(42)

# define run id and parameters to store session
run_id = str(time.time()).replace('.','')[:14]
model_name = 'rnn_cnn_2_' + run_id
model_file = model_name + '.h5'

# importing dataset
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.33,
                                                  random_state=42)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Define callback for early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mc = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

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
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# print(classifier.summary())

# Fitting the RNN to the training set
history = classifier.fit(x=X_train,
                         y=y_train,
                         validation_data=(X_val, y_val),
                         epochs=1,
                         batch_size=1024,
                         callbacks=[es, mc])


# def build_classifier():
#     classifier = Sequential()
#     classifier.add(Bidirectional(LSTM(n_lstm,
#                                       input_shape=X_train.shape[1:],
#                                       return_sequences=True)))
#     classifier.add(Conv1D(filters=16,
#                           kernel_size=3,  # should correspond to trigrams
#                           strides=1,
#                           activation='relu',
#                           input_shape=(max_len, n_lstm)))
#     classifier.add(MaxPooling1D(pool_size=2))
#     classifier.add(Bidirectional(LSTM(32,
#                                       input_shape=(
#                                       (max_len - 2) / 2, 16))))  # ((max_len - kernel_size + 1)/pool_size, filters)
#     classifier.add(Dense(output_dim,
#                          activation='softmax'))
#     opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#     classifier.compile(optimizer=opt,  # rmsprop usually good for RNN
#                        loss='binary_crossentropy',
#                        metrics=['accuracy'])
#     return classifier
#
#
# classifier = KerasClassifier(build_fn=build_classifier, batch_size=1024, nb_epoch=2)
# accuracies = cross_val_score(estimator=classifier,
#                              X=X_train,
#                              y=y_train,
#                              cv=10,
#                              n_jobs=-1)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
plt.savefig(model_name + '_accuracy.png')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
plt.savefig(model_name + '_loss.png')
plt.close()

# load best model
saved_model = load_model(model_file)

# Predict validation set results
y_pred = saved_model.predict(X_val)
y_pred = y_pred == np.amax(y_pred, axis=1, keepdims=True).astype('float64')
y_pred_binary = [el[0] for el in y_pred]
y_val_binary = [el[0] for el in y_val]
f1_val = f1_score(y_val_binary, y_pred_binary)
acc_val = accuracy_score(y_val_binary, y_pred_binary)

# Predict the test set results
y_pred = saved_model.predict(X_test)
y_pred = y_pred == np.amax(y_pred, axis=1, keepdims=True).astype('float64')
y_pred_binary = [el[0] for el in y_pred]
y_test_binary = [el[0] for el in y_test]
f1_test = f1_score(y_test_binary, y_pred_binary)
acc_test = accuracy_score(y_test_binary, y_pred_binary)

print('VAL. SET accuracy: {}'.format(acc_val))
print('VAL. SET f1-score: {}'.format(f1_val))

print('TEST SET accuracy: {}'.format(acc_test))
print('TEST SET f1-score: {}'.format(f1_test))
