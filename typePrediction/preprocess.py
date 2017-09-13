import pandas as pd
import config
import re


def getNumbersOfClasses():
    source_df = pd.read_csv(config.RAW_DATA_PATH+"/types.tsv", sep='\t')
    types =[]
    for index,row in source_df.iterrows():
        type = row['Type']
        types.append(type)
    typeslst = list(sorted(set(types)))
    print("sorted set of types: {}".format(typeslst))
    print("len of types: "+ str(len(types)))
    print("len of typesLst: "+ str(len(typeslst)))
    # len
    # of
    # types: 26530
    # len
    # of
    # typesLst: 1362
def removeBetweenWord(type):
    pattern = r'<(.*?)>'
    between_word = re.search(pattern, type).group(1)
    type = type.replace(between_word, "")
    return type

def preprocessTypes():
    # remove between <>

    source_df = pd.read_csv(config.RAW_DATA_PATH + "/types.tsv", sep='\t')

    for index, row in source_df.iterrows():
        type = row['Type']
        if "<" in type:
            cleaned = removeBetweenWord(type)
            row['Type'] = cleaned
            print(row['Type'])
    #removing duplicates
    source_df=source_df.drop_duplicates(subset=['Type', 'Name'])
    #write it back to a .csv
    source_df.to_csv(config.PROCESSED_DATA_DIR + "/preprocessed_types.tsv", index=False, sep='\t')

if __name__ == '__main__':
    # getNumbersOfClasses()
    preprocessTypes()