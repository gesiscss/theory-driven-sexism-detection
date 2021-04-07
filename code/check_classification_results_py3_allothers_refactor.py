#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pymongo import MongoClient
import bson
client = MongoClient()
db = client.dassexist
with open('/home/indira/mng.db') as f:
    pwd = f.read().strip()
db.authenticate('indira', pwd, source='login')


# In[2]:


from jha2017_preprocessing import preprocess, preprocess_jha2017, preprocess_light


# In[3]:


from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# In[4]:


from sklearn.svm import SVC


# In[5]:


from config import DATA_ROOT, GENDERED_VOCAB_REL_PATH


# In[6]:


from collections import defaultdict


# In[7]:


import os

import re


# In[8]:


import numpy as np
import pandas as pd
# get_ipython().magic(u'pylab inline')


# In[9]:


import seaborn as sns


# In[10]:


import unidecode
import pickle


# In[11]:


def compute_best_thold(X, y):
    tp, fp, tholds = roc_curve(y, X)

    min_distance_01 = np.infty
    min_distance_01_idx = np.nan
    for idx, point in enumerate(zip(tp, fp)):
        distance_from_01 = np.sqrt((point[0])**2 + (1- point[1])**2) #take the point that is closest to (0, 1), maximize acc
        if distance_from_01 < min_distance_01:
            min_distance_01 = distance_from_01
            min_distance_01_idx = idx

    best_thold = tholds[min_distance_01_idx]
    return best_thold
class ThresholdClassifier(BaseEstimator):
    def fit(self, X, y):
        self.threshold = compute_best_thold(X, y)
        return self
    def predict(self, X):
        return list(map(lambda x: x>=self.threshold, X))
        


# In[12]:


class JhaPreprocessor(TransformerMixin):
    def __init__(self, fix_encoding=False):
        self.fix_encoding = fix_encoding
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return preprocess_jha2017(X, self.fix_encoding)


# In[13]:


class LightPreprocessor(TransformerMixin):
    def __init__(self, fix_encoding=False):
        self.fix_encoding = fix_encoding
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return preprocess_light(X, self.fix_encoding) 


# In[14]:


class IndiraPreprocessor(TransformerMixin):
        
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        import re
        import string
        from nltk.corpus import stopwords
        stops = set(stopwords.words('english'))
        stops.discard('not')

        def strip_links(text):
            link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
            links         = re.findall(link_regex, text)
            for link in links:
                text = text.replace(link[0], ', ')    
            return text

        def strip_all_entities(text):
            entity_prefixes = ['@','#']
            for separator in  string.punctuation:
                if separator not in entity_prefixes :
                    text = text.replace(separator,' ')
            words = []
            for word in text.split():
                word = word.strip()
                if word:
                    #print word
                    if word[0] not in entity_prefixes and word not in stops:
                        words.append(word.lower())
                    elif word in stops:
                        continue
                    else:
                        words.append('UNK')
            return ' '.join(words)
        return list(strip_all_entities(strip_links(t)) for t in X)


# In[15]:


bert_version = "minprocessed_lower"    
with open('BERT_encoded_data/original_data_%s.pkl' %bert_version, 'rb') as f:
    og_bert = pd.DataFrame(pickle.load(f, encoding='latin1')   )
with open('BERT_encoded_data/modified_data_%s.pkl' %bert_version, 'rb') as f:
    mods_bert = pd.DataFrame(pickle.load(f, encoding='latin1')  )
bert_encodings = pd.concat((og_bert, mods_bert), ignore_index=True).set_index('_id').to_dict()['BERT_encoding']
bert_encodings = defaultdict(lambda: np.zeros_like(list(bert_encodings.values())[0]), 
                             bert_encodings)
class BertEncoder(TransformerMixin):
    def __init__(self, precomputed_encodings=bert_encodings):
        if precomputed_encodings:
            self.encodings = precomputed_encodings
    def fit(self, X, y=None):
        #if not trained, recompute them, optionally storing them
        return self
    def transform(self, X, y=None):
        return list(self.encodings[idx] for idx in X)


# In[16]:


with open(os.path.join(DATA_ROOT, GENDERED_VOCAB_REL_PATH)) as f:
    gender_words = set([line.strip() for line in f.readlines()])

gender_words = set(map(lambda x: x.replace("_", " "), gender_words))

gender_words_re = re.compile(r"\b(" +"|".join(map(re.escape, gender_words))+r")\b")

def has_gender_words(line, gender_re=gender_words_re):
    preprocessed_line = unidecode.unidecode(line).lower()
    return len(re.findall(gender_words_re, preprocessed_line))>0

class GenderWordClassifier(BaseEstimator):
    def __init__(self, gender_re=gender_words_re):
        self.gender_re=gender_re
    def fit(self, X, y):
        return self
    def predict(self, X):
        return list(map(lambda x: has_gender_words(x, self.gender_re), X))


# In[17]:


from bert_wrapper import FinetunedBertClassifier

from cnn_wrapper_noflags import CharCNN


# In[18]:


def model_factory(model='logit'):
    if model=='logit':
        from nltk.corpus import stopwords
        stops = set(stopwords.words('english'))
        stops.discard('not')
        
#         stops.discard('more')
#         stops.discard('as')

        return Pipeline(steps=[
            ('preprocess',
             JhaPreprocessor()),
            ('tfidf', 
            TfidfVectorizer(stop_words=stops)), 
           ('logit',
            LogisticRegression(solver='lbfgs', n_jobs=-1))])
    elif model=='thold':
        return ThresholdClassifier()
    elif model=='baseline':
        return GenderWordClassifier()
    elif model=='bert_finetuned':
        return Pipeline(steps=[
            ('preprocess',
             IndiraPreprocessor()),
#              LightPreprocessor()),
           ('bert',
            FinetunedBertClassifier())])
    elif model=='bert':
        return Pipeline(steps=[
            ('encode',
             BertEncoder()),
           ('csv',
            SVC(gamma="auto", kernel='linear'))])
    elif model=='cnn':
        return Pipeline(steps=[
            ('preprocess',
             IndiraPreprocessor()),
           ('cnn',
            CharCNN())])


# In[19]:


#get the originals
def get_originals(datasets=['benevolent', 'hostile', 'other'], balance_classes=True, sample_size=None):

    aggregation = [
        {"$match":{'dataset':{"$in":datasets},
                  'sexism_type':{"$exists":True}}}, 
        {"$project":{"dataset":1,
                 "sexist": (datasets == ['scales']) and {"$eq":["$reverse_coded",False]} or ("$sexism_type.sexist_binary"),

#                 "sexist": "$sexism_type.sexist_binary",
                 "toxicity": "$perspective_api_scores.text_TOXICITY",
                 "text":1}},
                                   ]
    if sample_size:
        aggregation.append({ "$sample": { "size": sample_size } })
    
    originals = pd.DataFrame(db.sentence.aggregate(aggregation))
    originals.set_index("_id", inplace=True)

    #equalize class sizes
    if balance_classes:
        min_class_size = originals.groupby("sexist").size().min()
        originals = originals.groupby("sexist").apply(lambda x:x.sample(n=min_class_size, replace=False)).reset_index(0, drop=True)
    print( "n originals")
    print( originals.groupby("sexist").size())
    print( )
    return originals


# In[ ]:


#get the originals
def get_bh_and_all_o(balance_classes=True, sample_size=None):
    aggregation = [
        {"$match":
             {'$or':
                 [{'dataset':{"$in":['benevolent', 'hostile']},
                      'sexism_type':{"$exists":True}},
                  {"dataset":'other'}
                 ]
             }
        }, 
        {"$project":{"dataset":1,
                 "sexist": "$sexism_type.sexist_binary",
                 "toxicity": "$perspective_api_scores.text_TOXICITY",
                 "text":1}},
                                   ]
    if sample_size:
        aggregation.append({ "$sample": { "size": sample_size } })
    
    originals = pd.DataFrame(db.sentence.aggregate(aggregation))
    originals.fillna({'sexist':False}, inplace=True)
    originals.set_index("_id", inplace=True)

    #equalize class sizes
    if balance_classes:
        min_class_size = originals.groupby("sexist").size().min()
        originals = originals.groupby("sexist").apply(lambda x:x.sample(n=min_class_size, replace=False)).reset_index(0, drop=True)
    print( "n originals")
    print( originals.groupby("sexist").size())
    print( )
    return originals


# In[20]:


#get the modifications
def get_modifications(originals):
    modifications = pd.DataFrame(db.modification.aggregate([{"$match":{'of_id':{"$in":originals[originals.sexist==True].index.tolist()},
                  'sexism_type.sexist_binary':False}}, 
        {"$project":{"dataset":1,
                 "sexist": "$sexism_type.sexist_binary",
                 "toxicity": "$perspective_api_scores.text_TOXICITY",
                 "of_id":1,
                 "text":1}}]))
    modifications.set_index("_id", inplace=True)
    modifications = modifications.groupby("of_id").apply(lambda x:x.sample(n=1)).reset_index(0, drop=True)
    print( "n modifications")
    print( len(modifications))
    print()
    return modifications


# In[21]:


#equalize sizes of the modifications and originals
def drop_originals_without_modifications_inplace(originals, modifications):

    missing_modification_filter = (~originals.index.isin(modifications.of_id.unique()))& originals.sexist
    originals.drop(originals[missing_modification_filter].index,
                  inplace = True)
    originals.drop(originals[~originals.sexist].sample(n=missing_modification_filter.sum(), replace=False).index, 
                  inplace = True)
    print( "n originals after dropping missing modifications")
    print( originals.groupby("sexist").size())
    print()


# In[22]:


#split data
def get_one_split(originals, modifications, test_frac = .3):

    originals_test = originals.groupby("sexist").apply(lambda x:x.sample(frac=test_frac,
                                                                          replace=False)).reset_index(0, drop=True)
    originals_train = originals[~originals.index.isin(originals_test.index)].sample(frac=1.)

    originals_test.groupby("sexist").size()

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


# In[23]:


# print( "dataset: bho")
# # originals_reproduction = get_originals(datasets=['benevolent', 'hostile', ])
# originals_reproduction = get_originals(datasets=['benevolent', 'hostile', 'other'])
# modifications_reproduction = get_modifications(originals_reproduction)
# drop_originals_without_modifications_inplace(originals_reproduction, modifications_reproduction)

# print( "dataset: callme")
# originals_replication = get_originals(datasets=['callme'])
# modifications_replication = get_modifications(originals_replication)
# drop_originals_without_modifications_inplace(originals_replication, modifications_replication)

# print( "dataset: scales")
# originals_goldstandard = get_originals(datasets=['scales'], balance_classes=False)



# print( "dataset: bh")
# originals_bh = get_originals(datasets=['benevolent', 'hostile', ])
# modifications_bh = get_modifications(originals_bh)
# drop_originals_without_modifications_inplace(originals_bh, modifications_bh)

# print( "dataset: scales train")
# originals_goldtrain = get_originals(datasets=['scales'], balance_classes=True)
# modifications_goldtrain = pd.DataFrame(columns=modifications_bh.columns) #empty df


# In[24]:


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


# In[25]:


# print( "dataset: omnibus")
# originals_omnibus = pd.concat((stratified_sample_df(originals_reproduction, "sexist", 300),
#                                stratified_sample_df(originals_replication, "sexist", 100),
#                                stratified_sample_df(originals_goldstandard, "sexist", 100),
#                               ))
# modifications_omnibus = get_modifications(originals_omnibus)
# # drop_originals_without_modifications_inplace(originals_omnibus, modifications_omnibus)

# originals_omnibus.groupby("dataset").size()

# originals_omnibus.groupby("sexist").size()


# In[26]:


results=list()


# In[ ]:


# split the originals
n_iterations = 20

for iteration_n in range(n_iterations):
    print("iteration", iteration_n)
    
    print( "dataset: bho")
#    originals_reproduction = get_originals(datasets=['benevolent', 'hostile', 'other'])
    originals_reproduction = get_bh_and_all_o()
    modifications_reproduction = get_modifications(originals_reproduction)
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
#                      "noother":{"original":originals_train_bh.reset_index(),
#                                     'adversarial':modifications_train_bh.reset_index()
#                                    },
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
#                      "noother":{"original":originals_test_bh.reset_index(),
#                                     'adversarial':modifications_test_bh.reset_index()
#                                    },
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
#                                              ('bert', "_id"), #
                                             ('cnn', "text"),
                                             ('bert_finetuned', "text"),
                                             ("thold","toxicity"), ('baseline', "text")]:
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
                                "y_test_ids": data_test._id.values
                                       })


# In[ ]:


originals_train_omnibus.groupby(['dataset', 'sexist']).size()


# In[ ]:


modifications_train_omnibus.groupby(['dataset', 'sexist']).size()


# In[ ]:


results_df = pd.DataFrame(results)


# In[ ]:


results_df.head()


# In[ ]:


def compute_scores(row):
    y_true, y_pred = row.y_test, row.y_pred
    classification_dict =  classification_report(y_true, y_pred, output_dict=True)
#     print(classification_dict)
#     for class_name, result_dict in classification_dict.items():
#         print(result_dict)
        
    classification_dict = {" ".join(((class_name=="True"and "sexist") or class_name == "False" and "nonsexist" or class_name,
                                    metric)) : value
                           for class_name, result_dict in filter(lambda x: x[0] in ["True", "False", "macro avg"],
                                                                 classification_dict.items())
                           for metric, value in result_dict.items()
#                            if class_name in ["True", "False", "macro avg"]
                          }
    return pd.Series(classification_dict)


# In[ ]:


metrics_df = results_df.apply(compute_scores, axis=1)


# In[ ]:


results_df = pd.concat((results_df, metrics_df), axis=1)


# In[ ]:


# results_df['train'] = results_df['train_domain'] + '_'
results_df['train'] =results_df['train_type']
results_df['train'] +='_'
results_df['train'] +=results_df['model_name']

results_df['test'] = results_df['test_domain'] + '_'
results_df['test'] +=results_df['test_type']


# In[83]:


# with open("../sexist_data/run_results_test.pkl", 'wb+') as f:
#     pickle.dump(results_df, f)


# In[ ]:


with open("../sexist_data/run_results_allothers_final2_refactor.pkl", 'wb+') as f:
    pickle.dump(results_df, f)

with open("../sexist_data/run_results_allothers_final2_refactor.pkl2", 'wb+') as f:
    pickle.dump(results_df, f, protocol=2)

# In[ ]:





# In[ ]:





# # In[123]:


# with open("../sexist_data/run_results_test.pkl", 'rb') as f:
#     results_df = pickle.load(f)


# # In[176]:


# with open("../sexist_data/cnn_results_test.pkl", 'rb') as f:
#     results_cnn_df = pickle.load(f)

# results_df = pd.concat((results_df,results_cnn_df), sort=False, ignore_index=True)


# # In[191]:


# with open("../sexist_data/finetuned_results_test.pkl", 'rb') as f:
#     results_finetuned_df = pickle.load(f)

# results_df = pd.concat((results_df,results_finetuned_df), sort=False, ignore_index=True)


# # In[192]:


# with open("../sexist_data/run_results_test.pkl2", 'wb+', ) as f:
#     pickle.dump(results_df, f, protocol=2)


# # In[193]:


# results_df.model_name.unique()


# # In[179]:


# results_df.train.unique()


# # In[180]:


# results_df[results_df.train.isin(['original_logit', 'original_thold', 'original_baseline',
#        'adversarial_logit', 'original_cnn',
#        'adversarial_cnn'])].groupby(["train", "test", 'train_domain'])[u'macro avg f1-score'].mean()


# # In[125]:


# results_df.columns


# # In[194]:


# #with vs. without others: same patterns, without just performs worse
# results_df[results_df.train_domain.isin(['noother', 'reproduction',
#                                         ]) & results_df.model_name.isin(['logit',"bert", 'cnn', 'bert_finetuned'
#                                         ]) & (results_df.train_domain == results_df.test_domain) ].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[202]:


# #with vs. without others: without just performs worse also outside. works worse overall on scales
# results_df[results_df.train_domain.isin(['noother', 'reproduction',
#                                         ]) & results_df.model_name.isin(['bert','logit', 'cnn'
#                                         ])& ~results_df.test_domain.isin(['noother', 'reproduction',
#                                         ])].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[196]:


# #it is possible to understand scales. bert gets ~70% in domain f1
# results_df[results_df.train_domain.isin(['goldtrain', ]) & results_df.test.isin(['goldtrain_original',
#                                         ])].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[197]:


# #training on scales does not help much on other datasets
# #only advantage is on omnibus, likley becasue it includes scales there as well
# results_df[results_df.train_domain.isin(['goldtrain', ]) & ~results_df.test.isin(['goldtrain_original','gold_original'
#                                         ])& ~results_df.model_name.isin(['baseline', 'thold'])].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[198]:


# #omnibus works best on scales; adversarial omnibus even better (68 sv 72 in-domain) --update w. finetuning original bert 69
# results_df[results_df.test.isin(['goldtrain_original', ]) & ~results_df.model_name.isin(['baseline', 'thold'])].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[199]:


# #omnibus performs nicely everywhere
# results_df[results_df.train_domain.isin(['omnibus', ]) & ~results_df.model_name.isin(['baseline', 'thold'])].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[189]:


# #cnn works... meh
# results_df[results_df.model_name.isin(['cnn', ]) & ~results_df.test.isin(['goldtrain_original',
#                                         ])& ~results_df.model_name.isin(['baseline', 'thold'])].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[200]:


# results_df[results_df.train_domain.isin(['omnibus', ]) & ~results_df.test.isin(['goldtrain_original',
#                                         ])& ~results_df.model_name.isin(['baseline', 'thold'])].groupby(["test","train", 'train_domain',
#                                                     ])[u'macro avg f1-score'].mean()


# # In[91]:


# results_df.train.unique()


# # In[66]:


# for idx, data in results_df[results_df.train.isin(['adversarial_logit', 'adversarial_bert', 
#        'original_logit', 'original_bert',
#        'original_thold', 'original_baseline'])].groupby(["train", "test", 'train_domain'])[u'macro avg f1-score'].mean().items():
#     print(idx,  data)


# # In[160]:


# toplot=results_df.copy()


# # In[161]:


# toplot = toplot.melt(id_vars=['train', 'test', 'train_domain'], value_vars = [u'macro avg f1-score',
# u'sexist recall',
#          u'sexist precision',])
# toplot.head()


# # In[162]:


# toplot.test.unique()


# # In[168]:


# toplot.train_domain.unique()


# # In[163]:


# toplot.train.unique()


# # In[164]:


# toplot.variable.unique()


# # In[165]:


# toplot = toplot[toplot.train.isin(['original_logit', 'adversarial_logit', 
#                                    'original_bert', 'adversarial_bert',
#                                    'original_thold','original_baseline', ])]

# toplot = toplot[toplot.test.isin([
#     'goldtrain_original',
#     'omnibus_original', 'omnibus_adversarial', 
#     'replication_original','replication_adversarial', 
#     'reproduction_original','reproduction_adversarial'])]

# toplot = toplot[toplot.train_domain.isin(['goldtrain', 'omnibus', 'replication', 'reproduction'])]


# # In[173]:


# #without others
# with plt.rc_context(dict(sns.axes_style("whitegrid"),
#                          **sns.plotting_context("notebook", font_scale=1.5))):
#     g = sns.catplot(x='test', y='value', hue='train', col='variable', row='train_domain',  
#                     data=toplot, kind='box',
#                     sharex=True, sharey=True,
#                     hue_order = ['original_logit', 'adversarial_logit', 
#                                    'original_bert', 'adversarial_bert',
#                                    'original_thold','original_baseline', ],
#                     col_order = [u'sexist precision', u'sexist recall', u'macro avg f1-score', ],
#                     row_order = ['reproduction','replication', 'goldtrain', 'omnibus', ],
#                     order = ['reproduction_original', 'reproduction_adversarial', 
#                              'replication_original', 'replication_adversarial', 
#                              'gold_original',
#                             'omnibus_original', 'omnibus_adversarial', ],
# # #                         linestyles=["--",'-', ":", "-.", ],
# #                     markers=['o', "D", 'v', 'x',],
# #                     legend=True
#                    )
#     for ax in g.axes[1]: ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
#     g.set_titles("{col_name}")
#     g.set_xlabels("test set")
# #     for ax, name in zip(g.axes[:, 0], ['reproduction', 'replication']): ax.set_ylabel(name)
# #     g.axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
# #           fancybox=True, shadow=True, ncol=5)
#     g.fig.tight_layout()

# #     g.despine()
    

# #     for hh, ls in zip(h, ["-",":", "-.", "-",":", "-.", ]):
# #         hh.set_linestyle(ls)
# #         hh.set_norm(None)
# #     plt.legend(handles=h, labels=l, handlelength=20)


# # In[ ]:


# toplot = toplot[toplot.train.isin(['original_logit', 'original_thold', 'original_baseline',
#        'adversarial_logit',])]


# # In[ ]:


# #without others
# with plt.rc_context(dict(sns.axes_style("whitegrid"),
#                          **sns.plotting_context("notebook", font_scale=1.5))):
#     g = sns.catplot(x='test', y='value', hue='train', col='variable', row='train_domain',  
#                     data=toplot, kind='point', sharex=True, sharey=True,
#                     hue_order = ['original_logit', 'adversarial_logit', 'original_baseline', 'original_thold'],
#                     col_order = [u'sexist precision', u'sexist recall', u'macro avg f1-score', ],
#                     row_order = ['reproduction', 'replication'],
#                     order = ['reproduction_original', 'reproduction_adversarial', 
#                              'replication_original', 'replication_adversarial', 
#                              'gold_original',],
# #                         linestyles=["--",'-', ":", "-.", ],
#                     markers=['o', "D", 'v', 'x',],
#                     legend=True
#                    )
#     for ax in g.axes[1]: ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
#     g.set_titles("{col_name}")
#     g.set_xlabels("test set")
#     for ax, name in zip(g.axes[:, 0], ['reproduction', 'replication']): ax.set_ylabel(name)
# #     g.axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
# #           fancybox=True, shadow=True, ncol=5)
#     g.fig.tight_layout()

# #     g.despine()
    

# #     for hh, ls in zip(h, ["-",":", "-.", "-",":", "-.", ]):
# #         hh.set_linestyle(ls)
# #         hh.set_norm(None)
# #     plt.legend(handles=h, labels=l, handlelength=20)


# # In[298]:


# thetale = results_df[~(#results_df.model_name.isin(['baseline', 'thold'])|\
#              results_df.train_domain.isin(['noother', 'goldtrain'])|\
#              results_df.test_domain.isin(['noother']))
#           ].groupby([ 'test_domain', 'test_type', 'train_domain', 'train_type','model_name', ])[
#     ['macro avg f1-score', 'sexist recall', 'sexist precision']].mean().reset_index(-1)


# # In[299]:


# thetale.columns=['model_name', 'F1', 'P(s)', 'R(s)']


# # In[300]:


# thetale.reset_index(inplace=True)


# # In[302]:


# thetale.model_name.unique()


# # In[303]:


# thetale['test_domain'] = thetale.test_domain.map(dict(zip(['goldtrain', 'omnibus', 'replication', 'reproduction'], 
#                                            ['scales', 'bhocs', 'callme', 'bho'])))
# thetale['train_domain'] = thetale.train_domain.map(dict(zip(['goldtrain', 'omnibus', 'replication', 'reproduction'], 
#                                            ['scales', 'bhocs', 'callme', 'bho'])))
# thetale['test_type'] = thetale.test_type.map(dict(zip(['adversarial', 'original'], 
#                                            ['M', 'O'])))
# thetale['train_type'] = thetale.train_type.map(dict(zip(['adversarial', 'original'], 
#                                            ['M', 'O'])))
# thetale['model_name'] = thetale.model_name.map(dict(zip(['baseline', 'bert', 'bert_finetuned', 'cnn', 'logit', 'thold'], 
#                                            ['gender word', 'bert', 'bert finetuned', 'cnn', 'logit', 'toxicity'])))


# # In[304]:


# thetale = thetale.melt(id_vars = [ 'test_domain', 'test_type', 'train_domain', 'train_type','model_name', ],
#              value_vars=['F1', 'P(s)', 'R(s)'],            )


# # In[305]:


# thetale = thetale.pivot_table(index=[ 'test_domain', 'test_type', 'train_domain', 'train_type',], columns=['model_name', 'variable']).round(2)


# # In[306]:


# thetale.index.rename(['test', 'type', 'train', 'type'], inplace=True)


# # In[307]:


# thetale.columns.rename([None, 'model', None], inplace=True)


# # In[308]:


# print( thetale.to_latex(multirow=True))


# # In[ ]:




