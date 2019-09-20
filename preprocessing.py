# Scope of this script is to create a dataset in the format suitable for training the models

from utilities.Preprocess import DataEncoder
import os.path as path
import pandas as pd
import numpy as np

DATA_FOLDER = '../../data/quora-insincere-questions-classification'
TRAIN_DATA = path.join(DATA_FOLDER, 'train.csv')


data = pd.read_csv(TRAIN_DATA)
data = data.sample(frac=0.1, random_state=42)

data_copy = data.copy()
data_train = data_copy.sample(frac=0.8, random_state=42)
data_test = data_copy.drop(data_train.index)

del data

preprocessor = DataEncoder(max_len=148,
                           column_name_sents='question_text',
                           column_name_labels='target')  # using max_len given by 95th percentile of question lengths

preprocessor.fit(data_train)
X_train, y_train = preprocessor.transform(data_train)

# store numpy arrays to file
np.save('X_train', X_train)
print('Stored X_train in file')
np.save('y_train', y_train)
print('Stored y_train in file')

del X_train, y_train

X_test, y_test = preprocessor.transform(data_test)

# store numpy arrays to file
np.save('X_test', X_test)
print('Stored X_test in file')
np.save('y_test', y_test)
print('Stored y_test in file')
