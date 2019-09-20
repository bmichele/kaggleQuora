from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class DataEncoder:
    def __init__(self, max_len, column_name_sents, column_name_labels):
        self.column_sents = column_name_sents
        self.column_labels = column_name_labels
        self.tokenizer = Tokenizer(filters='', char_level=True)
        self.one_hot_encoder_y = OneHotEncoder()
        self.one_hot_encoder_x = OneHotEncoder()
        self.max_len = max_len

    def fit(self, dataset):
        sentences = dataset[self.column_sents].values
        self.tokenizer.fit_on_texts(sentences)
        sequences = self.tokenizer.texts_to_sequences(sentences)
        sequences = pad_sequences(sequences, maxlen=self.max_len, truncating='pre', padding='pre')
        self.one_hot_encoder_x.fit(sequences.flatten().reshape(-1, 1))

        y = dataset[self.column_labels].values
        self.one_hot_encoder_y.fit(y.reshape([y.shape[0], 1]))

    def transform(self, dataset):
        sentences = dataset[self.column_sents].values
        sequences = self.tokenizer.texts_to_sequences(sentences)
        sequences = pad_sequences(sequences, maxlen=self.max_len, truncating='pre', padding='pre')
        x = self.one_hot_encoder_x.transform(sequences.flatten().reshape(-1, 1)).toarray()
        x = x.reshape((*sequences.shape, x.shape[1]))

        y = dataset[self.column_labels].values
        output_dim = len(set(y))
        y = self.one_hot_encoder_y.transform(y.reshape([y.shape[0], 1])).toarray()
        # print some stats
        print('number of unique states:', output_dim)
        print('total chars:', x.shape[2] - 1)  # take out one corresponding to the padding character
        lengths = [len(sent) for sent in sentences]
        print('max, min, average number of chars per sentence:', max(lengths), min(lengths), np.average(lengths))
        return x, y


# # TESTING
# ##########
# import pandas as pd
#
# dataset = pd.DataFrame({'queries': ['just pippo',
#                                     'only pippo',
#                                     'minni and clara',
#                                     'clara plus minni'],
#                         'labels': ['pippo',
#                                    'pippo',
#                                    'minni',
#                                    'minni']})
# dataset_test = pd.DataFrame({'queries': ['test pippo', '1234'], 'labels': ['pippo', 'minni']})
#
# preprocessor = DataEncoder(max_len=15, column_name_sents='queries', column_name_labels='labels')
# preprocessor.fit(dataset)
#
# X, y = preprocessor.transform(dataset)
# X_test, y_test = preprocessor.transform(dataset_test)
