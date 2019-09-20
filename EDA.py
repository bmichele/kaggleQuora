# Just a quick look at the dataset #
####################################

import pandas as pd
import os.path as path
import numpy as np

DATA_FOLDER = '../../data/quora-insincere-questions-classification'
TRAIN_DATA = path.join(DATA_FOLDER, 'train.csv')
TEST_DATA = path.join(DATA_FOLDER, 'test.csv')

data_train = pd.read_csv(TRAIN_DATA)
data_test = pd.read_csv(TEST_DATA)

print('Number of questions in train dataset is {}'.format(data_train.shape[0]))
print('Fields given for each question are {}'.format(data_train.columns.values))

lengths = [len(question) for question in data_train.question_text.values]
print('\nSome stats on question lengths\n')
len_stats = {
    'min_len': min(lengths),
    'max_len': max(lengths),
    'average': np.mean(lengths),
    'median': np.median(lengths),
    '95th percentile': np.percentile(lengths, 95)
}

for quantity, value in len_stats.items():
    print(quantity, value)

print('Number of 0 - 1 targets (0: good, 1: bad)')
print(data_train.target.value_counts())
