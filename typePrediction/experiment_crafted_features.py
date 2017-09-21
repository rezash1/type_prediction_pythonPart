


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from keras.wrappers.scikit_learn import KerasRegressor
import json
from sklearn.metrics.pairwise import cosine_similarity
import config
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

def precision(y_true, y_pred):
    return average_precision_score(y_true, y_pred)
def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)
def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def combine_features( feature_list ):
    return np.concatenate( (feature_list), axis=1 )


def get_features():
    loaded_feature_list = []

    for feature_name in config.features_to_use:
        print('Loading ', feature_name)
        filename = config.DUMPED_VECTOR_DIR + feature_name + '.pkl'

        loaded_feature = joblib.load( filename )

        if not isinstance(loaded_feature, np.ndarray):
            loaded_feature = loaded_feature.toarray()
        print('Shape =  {}, type = {}'.format(loaded_feature.shape, type(loaded_feature)))
        loaded_feature_list.append( loaded_feature )
        print('---------------------------------------------')

    return combine_features(loaded_feature_list)



def classify_train_test(X, y):
    y = np.array(y)
    print('Inside classify_train_test (X,y) ',X.shape, y.shape)

    ids = [i for i in range(len(X))]
    print(ids[:10])
    print(y[:10])
    y_train, y_test, id_train, id_test = train_test_split(y, ids, test_size=0.2, random_state=0)
    # y_train, y_test, id_train, id_test =cross_val_score(y, ids)
    X_train, X_test = X[id_train], X[id_test]

    print('X train, test', X_train.shape, X_test.shape)
    print('Y train, test', y_train.shape, y_test.shape)

    # clf = OneVsRestClassifier(LogisticRegression(verbose=0))
    # clf = OneVsRestClassifier(LinearSVC(random_state=0))
    clf = LogisticRegression()
    # clf = LinearSVC()
    clf.fit(X_train, y_train)

    # sc = cross_val_score(clf, X, y, scoring='f1_micro', cv=5, verbose=1)
    print('Training finished')

    predictions = clf.predict(X_test)
    print(predictions.shape)

    print('pred, true_label: ')
    for i in range(20):
        print(predictions[i], y_test[i])
    print("y shapes:")
    print(y_test.shape)
    print(predictions.shape)
    print("f1 score - micro")
    print(f1_score(y_test, predictions, average='micro'))
    # print("precision: ")
    # print(precision(y_test, predictions))
    # print("recall: ")
    # print(recall(y_test, predictions))
    # print(f1_score(y_test, predictions, average='macro'))
    # print(f1_score(y_test, predictions, average='weighted'))


def main():
    from sklearn import preprocessing
    lb = preprocessing.LabelEncoder()
    print('Loading X')
    X = get_features()
    print(X.shape)

    print('Loading Y')

    y = pd.read_csv(config.PROCESSED_DATA_DIR+'preprocessed_types.tsv', sep='\t')
    y = np.array(list(y['Type']))
    print(y[:10])
    print(y.shape)
    y = lb.fit_transform(y)
    print("after transform: ")
    print(y[:10])
    print(y.shape)
    print(type(X), type(y), X.shape, y.shape)

    classify_train_test(X, y)

if __name__ == '__main__':
    main()