from jha2017_preprocessing import preprocess, preprocess_jha2017, preprocess_light




from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


from config import DATA_ROOT, GENDERED_VOCAB_REL_PATH



from collections import defaultdict




import os

import re




import numpy as np
import pandas as pd


import seaborn as sns



import unidecode
import pickle


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
        



class JhaPreprocessor(TransformerMixin):
    def __init__(self, fix_encoding=False):
        self.fix_encoding = fix_encoding
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return preprocess_jha2017(X, self.fix_encoding)





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





from bert_wrapper import FinetunedBertClassifier

from cnn_wrapper_noflags import CharCNN




def model_factory(model='logit'):
    if model=='logit':
        from nltk.corpus import stopwords
        stops = set(stopwords.words('english'))
        stops.discard('not')
        
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
           ('bert',
            FinetunedBertClassifier())])
    elif model=='cnn':
        return Pipeline(steps=[
            ('preprocess',
             IndiraPreprocessor()),
           ('cnn',
            CharCNN(checkpoint_dir='checkpoints/cnn_test'))])
