#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from itertools import zip_longest
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from math import log
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def read_train_test(typ, N_FOLD):
  train = pd.read_csv('./Data/MHCPanII/{}{}.txt'.format(typ,N_FOLD), sep="\t", header=None)
  train.columns = ['peptide', 'aff', 'hla']
  
  return train

def slicing(string):
  s = string
  t = iter(s)
  k = ','.join(a+b+c  for a,b,c in zip_longest(t, t, t, fillvalue=""))
  k = k.split(',')
  return k

def transform_hla_name(string):
  string = string.replace('*', '')
  string = string.replace(':', '')
  r = re.compile("([a-zA-Z]+)([0-9]+)")
  
  return r.match(string).group(1) + '_' + r.match(string).group(2)

def hla_name_cut(string, position = 2):
  if string.startswith('HLA'):
    string = string.split('-')
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    string[position] = r.match(string[position]).group(1) + '_' + r.match(string[position]).group(2)
    return string[position]
  elif (string.startswith('DRB') & position ==1):
    string = string.replace('_', '')
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    string = r.match(string).group(1) + '_' + r.match(string).group(2)
    return string
  else:
    return np.nan
  
def tovec(serie, vec_size, window):
  #serie = serie.apply(lambda x : slicing(str(x)))
  documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(serie)]
  model = Doc2Vec(documents, vector_size = vec_size, window = window, min_count=1, workers=12)
  return model

def rename_col(data, string):
  name = []
  for c in data.columns:
    name.append(string + str(c))

  return name 


# In[15]:


def start(train):
  df = pd.read_csv('./Data/exon_Sheet1.csv', header = 1)
  df['4digit'] = df['4digit'].apply(lambda x: transform_hla_name(x))
  df.rename(columns={'4digit':'hla'},inplace = True)
  for i in range(1, 7):
    name = 'exon'+ str(i)
    df[name] = df[name].apply(lambda x: slicing(str(x)))

  #train = read_train_test('train',1)
  #test = read_train_test('test',1)

  peptide = Doc2Vec.load('./Data/Doc2Vec')
  #train = pd.merge(train, df.drop(columns = '8digit'), on = 'hla', how = 'left')
  #train = train['hla'].drop_duplicates()
  #split DPA/DPB DQA/DQB
  train['hla_1'] = train['hla'].apply(lambda x: hla_name_cut(x, position = 1))
  train['hla_2'] = train['hla'].apply(lambda x: hla_name_cut(x))
  #merge train with exon
  df['hla_1'] = df['hla']; df['hla_2'] = df['hla']

  train = train[train['hla_1'].notna()]
  train = pd.merge(train, df.drop(columns = ['8digit', 'hla_2', 'hla']), on = 'hla_1', how = 'left')
  train = pd.merge(train, df.drop(columns = ['8digit', 'hla_1', 'hla']), on = 'hla_2', how = 'left')
  for c in ['exon1_y', 'exon2_y','exon3_y', 'exon4_y', 'exon5_y', 'exon6_y']:
    train[c] = train[c].apply(lambda d: d if isinstance(d, list) else [])
  train['exon'] = train['exon1_x']+train['exon2_x']+train['exon3_x']+train['exon4_x']+ train['exon5_x']+train['exon6_x']+train['exon1_y']+train['exon2_y']+train['exon3_y']+train['exon4_y']+train['exon5_y']+train['exon6_y']
  
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer(analyzer='word')
  a = train['exon'].apply(lambda x: ' '.join(c for c in x))

  import pickle
  vectorizer = pickle.load(open('./Data/HLA_TFIDF.pickle', 'rb'))
  X = np.array(vectorizer.transform(a).toarray(), dtype=np.float16)
  
  HLA_TFIDF = pd.DataFrame(X)
  name = []
  for i in range(len(vectorizer.get_feature_names())):
    name.append('HLA_' + str(vectorizer.get_feature_names()[i]) + '_tfidf')
  HLA_TFIDF.columns = name
  HLA_TFIDF['hla'] = train['hla']
  return HLA_TFIDF.drop_duplicates()
  


# In[ ]:




