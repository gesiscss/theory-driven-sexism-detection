# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pandas as pd
import os
import re
# import preprocessor
from bs4 import BeautifulSoup
import codecs
import numpy as np
from functools import partial

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
url_pattern = re.compile('''((https?:\/\/)?(?:www\.|(?!www))[a-zA-Z0-9]([a-zA-Z0-9-]+[a-zA-Z0-9])?\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})''', 
                         flags=re.UNICODE)
mention_pattern = re.compile('([^a-zA-Z0-9]|^)@\S+', flags=re.UNICODE)
hashtag_pattern = re.compile('([^a-zA-Z0-9]|^)#\S+', flags=re.UNICODE)
rt_pattern = re.compile('([^a-zA-Z0-9]|^)(rt|ht|cc)([^a-zA-Z0-9]|$)', flags=re.UNICODE)
def detweet(text):
    return re.sub(url_pattern, '', 
               re.sub(rt_pattern, '', 
                      re.sub(mention_pattern, '',
                             re.sub(hashtag_pattern, '', 
                                 re.sub(emoji_pattern, '', 
                                    text)))))
def normalize(text):
    return re.sub(r"\s+", " ", #remove extra spaces
                  re.sub(r'[^a-zA-Z0-9]', ' ', #remove non alphanumeric, incl punctuation
                         text)).lower().strip() #lowercase and strip
def fix_encoding_and_unescape(text):
    return BeautifulSoup(text.decode('unicode-escape')).get_text()
def preprocess(text, fix_encoding=False):
    if (type(text)==str) or (type(text)==unicode):
        if fix_encoding:
            return normalize(detweet(fix_encoding_and_unescape(text)))
        else:
            return normalize(detweet(text))
    else:
        return text

def preprocess_light(tweets, fix_encoding=False):
    def _preprocess_light(text, fix_encoding=False):
        if (type(text)==str) or (type(text)==unicode):
            if fix_encoding:
                return normalize(re.sub(url_pattern, '',fix_encoding_and_unescape(text)))
            else:
                return normalize(re.sub(url_pattern, '',text))
        else:
            return text
    return map(partial(_preprocess_light, fix_encoding=fix_encoding), tweets)
    
def preprocess_jha2017(tweets, fix_encoding = False):
    """Replicates preprocessing as per Jha et al. 2017. 
    It removes usernames, punctuations, emoticons, hyperlinks/URLs and RT tag.
    It also lowercases, which seems a sensible thing to do.
    params: 
        tweets: list of unicode strings

    returns: list of unicode strings
    """
    return map(partial(preprocess, fix_encoding=fix_encoding), tweets)
#     tweets_ = tweets.copy()
#     if fix_encoding:
        
#     return map(preprocess, tweets_)
#     df = pd.DataFrame(tweets, columns=['tweet'])
#     if fix_encoding:
#         df['tweet'] = df.tweet.apply(lambda x: x.decode('unicode-escape'))
#         unescape_html = lambda x: BeautifulSoup(x).get_text()
#         df['tweet'] = df.tweet.apply(unescape_html)
#     preprocessor.set_options(preprocessor.OPT.EMOJI, 
#                              preprocessor.OPT.HASHTAG, 
#                              preprocessor.OPT.MENTION,
#                              preprocessor.OPT.URL,
#                              preprocessor.OPT.SMILEY, 
#                              preprocessor.OPT.RESERVED
#                             )
#     strip_urls_mentions_hashtags_emojis_rt = lambda x: preprocessor.clean(x.encode('utf-8').strip())
#     df['tweet'] = df.tweet.apply(strip_urls_mentions_hashtags_emojis_rt) 
#     remove_punctuation = lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x)
#     df['tweet'] = df.tweet.apply(remove_punctuation)
#     remove_extra_spaces = lambda x: re.sub(r"\s+", " ", x).strip()
#     df['tweet'] = df.tweet.apply(remove_extra_spaces)
#     df['tweet'] = df.tweet.str.lower()
#     return df.tweet.values.tolist()
if __name__=='__main__':
    from config import DATA_ROOT, BENEVOLENT_SEXISM_REL_PATH, HOSTILE_SEXISM_REL_PATH
    tweets = None
    with codecs.open(os.path.join(DATA_ROOT, BENEVOLENT_SEXISM_REL_PATH), encoding='utf8') as file_handle:
        tweets = file_handle.readlines()
#     print preprocess_jha2017(tweets)
