from jha2017_preprocessing import preprocess, preprocess_jha2017, preprocess_light



from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline




from sklearn.svm import SVC



from config import DATA_ROOT, GENDERED_VOCAB_REL_PATH


from collections import defaultdict


import os

import re


import numpy as np
import pandas as pd


import seaborn as sns


import unidecode
import pickle


from utils import get_originals, get_modifications, drop_originals_without_modifications_inplace, stratified_sample_df, get_one_split
from classification_utils import model_factory




def compute_scores(row):
    y_true, y_pred = row.y_test, row.y_pred
    classification_dict =  classification_report(y_true, y_pred, output_dict=True)        
    classification_dict = {" ".join(((class_name=="True"and "sexist") or class_name == "False" and "nonsexist" or class_name,
                                    metric)) : value
                           for class_name, result_dict in filter(lambda x: x[0] in ["True", "False", "macro avg"],
                                                                 classification_dict.items())
                           for metric, value in result_dict.items()
#                            if class_name in ["True", "False", "macro avg"]
                          }
    return pd.Series(classification_dict)

results=list()


# split the originals
n_iterations = 5

for iteration_n in range(n_iterations):
    print("iteration", iteration_n)
    
    print("dataset: bho")
    originals_reproduction = get_originals(datasets=['benevolent', 'hostile', 'other'])

    modifications_reproduction = get_modifications(originals_reproduction)

    print(len(originals_reproduction), len(modifications_reproduction))

    drop_originals_without_modifications_inplace(originals_reproduction, modifications_reproduction)

    print( "dataset: callme")
    originals_replication = get_originals(datasets=['callme'])
    modifications_replication = get_modifications(originals_replication)
    drop_originals_without_modifications_inplace(originals_replication, modifications_replication)

    print( "dataset: scales complete")
    originals_goldstandard = get_originals(datasets=['scales'], balance_classes=False)

    print( "dataset: bh")
    originals_bh = get_originals(datasets=['benevolent', 'hostile', ])
    modifications_bh = get_modifications(originals_bh)
    drop_originals_without_modifications_inplace(originals_bh, modifications_bh)

    print( "dataset: scales train")
    originals_goldtrain = get_originals(datasets=['scales'], balance_classes=True)
    modifications_goldtrain = pd.DataFrame(columns=modifications_bh.columns) #empty df

    print(len(originals_reproduction), len(modifications_reproduction), len(originals_replication), len(modifications_replication))

 
    originals_train_replication, originals_test_replication, modifications_train_replication, modifications_test_replication = get_one_split(originals_replication, modifications_replication)
    originals_train_reproduction, originals_test_reproduction, modifications_train_reproduction, modifications_test_reproduction = get_one_split(originals_reproduction, modifications_reproduction)
    
    originals_train_bh, originals_test_bh, modifications_train_bh, modifications_test_bh = get_one_split(originals_bh, modifications_bh)
    
    originals_train_goldtrain, originals_test_goldtrain, _, _ = get_one_split(originals_goldtrain, modifications_goldtrain)

    originals_train_omnibus = pd.concat((stratified_sample_df(originals_train_reproduction, "sexist", 300),
                               stratified_sample_df(originals_train_replication, "sexist", 100),
                               stratified_sample_df(originals_train_goldtrain, "sexist", 100),
                              ),sort=False).sample(frac=1.)
    modifications_train_omnibus = pd.concat((stratified_sample_df(modifications_train_reproduction, "sexist", 300),
                               stratified_sample_df(modifications_train_replication, "sexist", 100),
                               stratified_sample_df(originals_train_goldtrain, "sexist", 100),
                              ),sort=False).sample(frac=1.)
#     modifications_train_omnibus = get_modifications(originals_train_omnibus)
    
    originals_test_omnibus = pd.concat((stratified_sample_df(originals_test_reproduction, "sexist", 300),
                               stratified_sample_df(originals_test_replication, "sexist", 100),
                               stratified_sample_df(originals_test_goldtrain, "sexist", 100),
                              ),sort=False)
    modifications_test_omnibus = pd.concat((stratified_sample_df(modifications_test_reproduction, "sexist", 300),
                               stratified_sample_df(modifications_test_replication, "sexist", 100),
                               stratified_sample_df(originals_test_goldtrain, "sexist", 100),
                              ),sort=False)
#     modifications_test_omnibus = get_modifications(originals_test_omnibus)
    

#     originals_train_omnibus, originals_test_omnibus, modifications_train_omnibus, modifications_test_omnibus = get_one_split(originals_omnibus, modifications_omnibus)

    training_sets = {
                     "replication":{"original":originals_train_replication.reset_index(),
                                     'adversarial':modifications_train_replication.reset_index()
                                    },
                     "reproduction":{"original":originals_train_reproduction.reset_index(),
                                    'adversarial':modifications_train_reproduction.reset_index()
                                   },
                     "omnibus":{"original":originals_train_omnibus.reset_index(),
                                    'adversarial':modifications_train_omnibus.reset_index()
                                   },
#                      "goldtrain":{"original":originals_train_goldtrain.reset_index()}
                    }
    test_sets = {"replication":{"original":originals_test_replication.reset_index(),
                                    'adversarial':modifications_test_replication.reset_index()
                                   },
                 "reproduction":{"original":originals_test_reproduction.reset_index(),
                                    'adversarial':modifications_test_reproduction.reset_index()
                                   },
                     "omnibus":{"original":originals_test_omnibus.reset_index(),
                                    'adversarial':modifications_test_omnibus.reset_index()
                                   },
#                      "gold":{"original":originals_goldstandard.reset_index()},
                     "goldtrain":{"original":originals_test_goldtrain.reset_index()}
                    }
    
    target_column = 'sexist'
    for train_domain in training_sets:
        for train_type, data_train in training_sets[train_domain].items():
            for model_name, input_column in [('logit', "text"), 
                                             ('cnn', "text"),
                                             ('bert_finetuned', "text"),
                                             ("thold","toxicity"),
                                             ('baseline', "text")]:
                model = model_factory(model_name)
                X_train, y_train = data_train[input_column].values, data_train[target_column].values
                print('training',model_name,'on', train_type,train_domain)
                model.fit(X_train, y_train)
                for test_domain in test_sets:
                    for test_type, data_test in test_sets[test_domain].items():
                        X_test, y_test = data_test[input_column].values, data_test[target_column].values
                        y_pred = model.predict(X_test)
                        results.append({"train_domain":train_domain,
                                "train_type":train_type,
                                "model_name":model_name,
                                "test_domain":test_domain,
                                "test_type":test_type,
                                "y_test": y_test,
                                "y_pred": y_pred,
                                "y_test_ids": data_test.id.values
                                       })



originals_train_omnibus.groupby(['dataset', 'sexist']).size()


modifications_train_omnibus.groupby(['dataset', 'sexist']).size()


results_df = pd.DataFrame(results)


results_df.head()


metrics_df = results_df.apply(compute_scores, axis=1)




results_df = pd.concat((results_df, metrics_df), axis=1)


# results_df['train'] = results_df['train_domain'] + '_'
results_df['train'] =results_df['train_type']
results_df['train'] +='_'
results_df['train'] +=results_df['model_name']

results_df['test'] = results_df['test_domain'] + '_'
results_df['test'] +=results_df['test_type']


with open("../results/all_runs_new_data.pkl", 'wb+') as f:
    pickle.dump(results_df, f)

with open("../results/all_runs_new_data.pkl2", 'wb+') as f:
    pickle.dump(results_df, f, protocol=2)
