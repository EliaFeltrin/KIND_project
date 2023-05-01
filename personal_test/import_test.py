# import data from ../dataset/KIND-main/dataset/

import pandas as pd
import string
import re

# Path to the dataset folder 
DATASET_PATH = '../dataset/KIND-main/dataset/'
TEST = 'test'
TRAIN = 'train'

def import_data(file_name, part):
    format = '.tsv'

    # Read the tab-separated file using pandas
    df = pd.read_csv(DATASET_PATH + file_name + '_' + part + format, sep='\t')

    return df

def add_column_names(df):
    return pd.DataFrame({'world': df.iloc[:, 0], 'label': df.iloc[:, 1]})

def to_lowerCase(df):
    return pd.DataFrame({'world': df['world'].str.lower(), 'label': df['label']})

def remove_nonEntity(df):
    all_worlds = df.iloc[:, 0].tolist()
    all_labels = df.iloc[:, 1].tolist()

    worlds = []
    labels = []

    for i in range(len(all_worlds)):
        if all_labels[i] != 'O':
            worlds.append(all_worlds[i])
            labels.append(all_labels[i])

    return pd.DataFrame({'world': worlds, 'label': labels})

def remove_punctuation(df):
    punct = string.punctuation
    all_worlds = df.iloc[:, 0].tolist()
    all_labels = df.iloc[:, 1].tolist()

    worlds = []
    labels = []

    for i in range(len(all_worlds)):
        if all(char in punct for char in all_worlds[i]) == False:
            worlds.append(all_worlds[i])
            labels.append(all_labels[i])

    return pd.DataFrame({'world': worlds, 'label': labels})

def vocab_size(df):
    return len(df['world'].unique())

def doc_length(df):
    return len(df)

def import_basicFormatting(file_name, part):
    df = import_data(file_name, part)
    df = add_column_names(df)
    df = to_lowerCase(df)

    return df

df = import_basicFormatting('degasperi', TRAIN)
no_punct_df = remove_punctuation(df)
entity_df = remove_nonEntity(no_punct_df)


print("ENTIRE DATASET: \n\tdoc_length: ", doc_length(df), "\n\tvocab_size: ",vocab_size(df), "\n")
print("NO PUNCTUATION: \n\tdoc_length: ", doc_length(no_punct_df), "\n\tvocab_size: ",vocab_size(no_punct_df), "\n")
print("ENTITY: \n\tdoc_length: ", doc_length(entity_df), "\n\tvocab_size: ",vocab_size(entity_df), "\n")