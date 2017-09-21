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


    # Unique types: 1362
def removeBetweenWord(type):
    pattern = r'<(.*?)>'
    between_word = re.search(pattern, type).group(0)
    type = type.replace(between_word, "")
    return type
def split(word):
    word = word.split("_")
    words = " ".join(word)
    print(words)
    return words
def convertCamelCase(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    finals = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    print(finals)
    return finals
def preprocessTypes():
    # remove between <>

    source_df = pd.read_csv(config.RAW_DATA_PATH + "/types.tsv", sep='\t')

    for index, row in source_df.iterrows():
        type = row['Type']
        name = row["Name"]
        if "<" in type:
            cleaned = removeBetweenWord(type)
            if "<" in cleaned:
                cleaned = cleaned.replace("<", "")
            if ">" in cleaned:
                cleaned = cleaned.replace(">", "")
            row['Type'] = cleaned
        print(name)
        row["Name"] = convertCamelCase(str(name))
        name = row["Name"]
        if "_" in str(name):
            print(name)
            row["Name"] = split(name)
            # print(row['Type'])
    #removing duplicates
    source_df=source_df.drop_duplicates(subset=['Type', 'Name'])
    #write it back to a .csv
    source_df.to_csv(config.PROCESSED_DATA_DIR + "/preprocessed_types.tsv", index=False, sep='\t')

if __name__ == '__main__':
    # getNumbersOfClasses()
    preprocessTypes()
