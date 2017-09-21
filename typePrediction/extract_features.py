
from features import lexical
import nltk
import config
import os
import traceback
from sklearn.externals import joblib
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import json









features_dict = dict(

    # n-gram
    unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=str.split,
                                         analyzer="word",
                                         stop_words='english', lowercase=True, min_df=1),
    bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=str.split,
                                        analyzer="word",
                                        lowercase=True, min_df=1),
    trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=str.split,
                                         analyzer="word",
                                         lowercase=True, min_df=1),

    binary_unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=nltk.word_tokenize,
                                         analyzer="word", use_idf=False, smooth_idf=False,
                                         stop_words='english', lowercase=True, min_df=1),
    binary_bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=nltk.word_tokenize,
                                        analyzer="word", use_idf=False, smooth_idf=False,
                                        lowercase=True, min_df=1),
    binary_trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=nltk.word_tokenize,
                                         analyzer="word", use_idf=False, smooth_idf=False,
                                         lowercase=True, min_df=1),

    #
    # #char ngramdeco
    char_tri=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), analyzer="char",
                                          lowercase=True, min_df=5),
    char_4_gram=lexical.NGramTfidfVectorizer(ngram_range=(4, 4), analyzer="char",
                                             lowercase=True, min_df=5),

    char_5_gram=lexical.NGramTfidfVectorizer(ngram_range=(5, 5), analyzer="char",
                                             lowercase=True, min_df=5)
)


class doc():
    def __init__(self, text):
        if isinstance(text, float):
            text = ''
        self.content = text



def load_data():
    """
    Will Change for each project
    :return:
    """
    print("LOADING DATA")
    document_list = []

    df = pd.read_csv(config.PROCESSED_DATA_DIR + 'preprocessed_types.tsv', sep='\t')
    for index, row in df.iterrows():
        document_list.append(
            doc(str(row['Name']))
        )

    print('===============================')

    return document_list


def extract_and_dump_features():
    """
    This method gets the list of active features in config.features_to_extract.
    Then it extract features with corresponding extractor and dumps to file.

    :return:
    """
    data = load_data()

    for feature_name in config.features_to_extract:
        try:
            print('Extracting Feature {}'.format(feature_name))
            extracted_feature = features_dict[feature_name].fit_transform(data)
            if isinstance(extracted_feature, list):
                extracted_feature = extracted_feature.toarray()
            print('Extraction Complete of {}'.format(feature_name))
            print('Shape = {}'.format(extracted_feature.shape))
            print("***")
            print(extracted_feature[:10])
            # print(extracted_feature)

            joblib.dump(extracted_feature, os.path.join(config.DUMPED_VECTOR_DIR, feature_name + '.pkl'))
            print('Feature {} vectors are dumped to {}'.format(feature_name,
                                                               os.path.join(config.DUMPED_VECTOR_DIR, feature_name +
                                                                            '.pkl')))
            print('=========================')

            # try:
            #     vocab = {k: str(v) for k, v in features_dict[feature_name].vocabulary_.items()}
            #
            #     with open(os.path.join(config.DUMPED_VECTOR_DIR, '_dict_' + feature_name + '.json'), 'w') as jf:
            #         json.dump(vocab, jf, ensure_ascii=True)
            #         jf.close()
            # except:
            #     pass
        except:
            print(print('FAILED Extracting Feature {}'.format(feature_name)))
            traceback.print_exc()


def extract_classes():
    from sklearn import preprocessing
    lb = preprocessing.LabelEncoder()

    df = pd.read_csv(config.PROCESSED_DATA_DIR + 'preprocessed_types.tsv', sep='\t')
    scores = []

    for index, row in df.iterrows():
        if str(row['Type']) == 'nan':
            print(row['Type'])
            print(index)
            break
        scores.append(row['Type'])
    print("Total: ")
    print(len(scores))
    # scores = lb.fit_transform(scores)
    # print(scores.shape)
    print(scores[:10])
    print("Set: ")
    print(len(set(scores)))

    joblib.dump(scores, os.path.join(config.DUMPED_VECTOR_DIR, 'true_classes.pkl'))

if __name__ == '__main__':
    extract_and_dump_features()
    extract_classes()