from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from config import DATA_ROOT, GENDERED_VOCAB_REL_PATH



from collections import defaultdict




import os

import re


import numpy as np
import pandas as pd


import unidecode


#get the originals
def get_originals(datasets=['benevolent', 'hostile', 'other'], balance_classes=True, sample_size=None):

     
    originals = pd.read_csv("../data/sexism_data/sexism_data.csv", index_col = False, encoding = 'utf-8')

    originals['sexist'] = originals['sexist'].fillna(False)
    originals = originals[originals['toxicity'].notna()]

    # drop modifications
    originals = originals[originals['of_id'] == -1]

    originals = originals[originals.dataset.isin(datasets)]

    #equalize class sizes
    if balance_classes:
        min_class_size = originals.groupby("sexist").size().min()
        originals = originals.groupby("sexist").apply(lambda x:x.sample(n=min_class_size,
         replace=False)).reset_index(0, drop=True)

    originals = originals.drop(['of_id'], axis = 1)

    return originals    



#get the modifications
def get_modifications(originals):
     
    modifications = pd.read_csv("../data/sexism_data/sexism_data.csv", index_col = False, encoding = 'utf-8')
    # only pick non-sexist modifications
    modifications = modifications[modifications['sexist'] == False]
    modifications = modifications[modifications['toxicity'].notna()]
    modifications = modifications[modifications['of_id'] != -1]
    modifications = modifications[modifications.of_id.isin(originals['id'].values)]
    modifications = modifications.groupby("of_id").apply(lambda x:x.sample(n=1)).reset_index(0, drop=True)

    return modifications 


#equalize sizes of the modifications and originals
def drop_originals_without_modifications_inplace(originals, modifications):

    
    originals.sexist = originals.sexist.astype(np.bool_)
    modifications.sexist = modifications.sexist.astype(np.bool_)
    modifications['of_id'] = modifications['of_id'].astype(int)
    originals['id'] = originals['id'].astype(int)


    missing_modification_filter = (~originals.id.isin(modifications.of_id.unique()))& originals.sexist
    originals.drop(originals[missing_modification_filter].index,
                  inplace = True)
    originals.drop(originals[~originals.sexist].sample(n=missing_modification_filter.sum(), replace=False).index, 
                  inplace = True)



#split data
def get_one_split(originals, modifications, test_frac = .3):

    try:
        originals.set_index("id", inplace=True)
    except:
        pass

    originals.sexist = originals.sexist.astype(np.bool_)
    modifications.sexist = modifications.sexist.astype(np.bool_)

    originals_test = originals.groupby("sexist").apply(lambda x:x.sample(frac=test_frac,
                                                                          replace=False)).reset_index(0, drop=True)
    originals_train = originals[~originals.index.isin(originals_test.index)].sample(frac=1.)


    # match the sexist in the test set
    tomodify_test = originals_test[originals_test.sexist].index
    modifications_test = pd.concat((originals_test[originals_test.sexist],
               modifications[modifications.of_id.isin(tomodify_test)]
              ), sort=False)

    # match half of the sexist in the training set
    tomodify_train = originals_train[originals_train.sexist].sample(frac=.5).index
    modifications_train = pd.concat((originals_train.drop(originals_train[~originals_train.sexist].sample(n=len(tomodify_train)).index),
               modifications[modifications.of_id.isin(tomodify_train)]
              ), sort=False).sample(frac=1.)
    return originals_train, originals_test, modifications_train, modifications_test


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


if __name__ == "__main__":
    print()