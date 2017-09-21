import os
import socket

# machine = socket.gethostname()
#
# machine = socket.gethostname()
# ROOT_DIR = os.path.dirname(__file__)
# ROOT_DIR = '/'.join(ROOT_DIR.split('/')[:-1]) + '/'
# print(machine)
# print(ROOT_DIR)

RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_DIR = 'data/preprocessed/'
RESULTS_DIR = 'data/outputs/'
DUMPED_VECTOR_DIR = 'data/vectors/'

#WORD_EMBEDDING_VECTOR_PATH = '/raid/data/skar3/GoogleNews-vectors-negative300.bin.gz'

features_to_extract = [
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    'char_tri',
    'char_4_gram',
    'char_5_gram',
    # 'pos',
    # 'verbTenses'
    # 'hasQuestion'
    # 'posPENN'
    # 'two_skip_3_grams',
    # 'two_skip_2_grams',
    # 'concepts',
    # 'stemmed_concepts',
    # 'google_word_emb',
    # 'cashtag',
    # 'source'
    # 'polarity',
    # 'sensitivity',
    # 'attention',
    # 'aptitude',
    # 'pleasantness',
    # 'stemmed_polarity',
    # 'stemmed_sensitivity',
    # 'stemmed_attention',
    # 'stemmed_aptitude',
    # 'stemmed_pleasantness',
    # 'company'

]

features_to_use = [
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    'char_tri',
    'char_4_gram',
    'char_5_gram',
    # 'pos',
    # # 'posPENN'
    # 'verbTenses'
    # 'hasQuestion'
    # 'char_5_gram',
    # 'two_skip_3_grams',
    # 'two_skip_2_grams',
    # 'concepts',
    # 'stemmed_concepts',
    # 'google_word_emb',
    # 'polarity',
    # 'sensitivity',
    # 'attention',
    # 'aptitude',
    # 'pleasantness',
    # 'stemmed_polarity',
    # 'stemmed_sensitivity',
    # 'stemmed_attention',
    # 'stemmed_aptitude',
    # 'stemmed_pleasantness',
    # 'company'
]
