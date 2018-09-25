##
# preprocess.py
#
# Script for reading the text data and extracting features
# Stores the result as a pickled pandas dataframe

import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

NEG_PATH = "data/rt-polarity.neg"
POS_PATH = "data/rt-polarity.pos"

PS = PorterStemmer()

def process_phrase(phrase, *functions):
    # Standardize english phrases with nltk
    words = phrase.split()

    for f in functions:
        words = map(f, words)

    return words

def remove_stopwords(phrase):
    return ' '.join([w for w in phrase.split() if w not in stopwords.words('english')])

def stem_words(phrase):
    return ' '.join([PS.stem(w) for w in phrase.split()])

def add_data(df, index, *functions):
    for f in functions:
        new_data = df.applymap(f)

    return pd.concat(new_data, axis="rows", keys=df.keys + [index])

def join_constant_to_df(constant, dataframe):
    # Appends a constant value series to the dataframe
    dataframe['class'] = pd.Series([constant] * dataframe.index.size, index = dataframe.index)

def apply_on_df_column(df, column, func):
    # Applies a function onto the series stored in one column. Returns new df
    applied_df = df.copy()

    applied_df[column] = applied_df[column].apply(func)
    return applied_df

# pandas.read_csv options specific to our CSV format 
csv_opts = {
    'header': None,
    'delimiter': '\n',
    'names': ['phrase'],
    'encoding': 'ISO-8859-1'
}

# 0-CLASSIFIED DATA
neg = pd.read_csv(NEG_PATH, **csv_opts)
join_constant_to_df(0, neg)

# 1-CLASSIFIED DATA
pos = pd.read_csv(POS_PATH, **csv_opts)
join_constant_to_df(1, pos)

raw_data = pd.concat([neg, pos], axis='rows')
stemmed_data = apply_on_df_column(raw_data, 'phrase', stem_words)
stopwords_data = apply_on_df_column(raw_data, 'phrase', remove_stopwords)
stemmed_stopwords_data = apply_on_df_column(stemmed_data, 'phrase', remove_stopwords)

from itertools import product
from sklearn.feature_extraction.text import CountVectorizer

#for ngram_range, data, min_df in product(*list(data_parameters.values())):
for ngram_range, (morph, data), min_df in product(
        ((1, 1), (1,2)),
        (
            ('raw', raw_data),
            ('stem', stemmed_data),
            ('stopwords', stopwords_data),
            ('stem&stopwords', stemmed_stopwords_data)
        ),
        (5, 25, 125)):

    cv = CountVectorizer(min_df=min_df, ngram_range=ngram_range)

    X = cv.fit_transform(data['phrase'].values).toarray()
    y = raw_data['class'].values

    print(ngram_range, morph, min_df)
    print(X)
