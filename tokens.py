#!/usr/bin/env python
# coding: utf-8

# # Токенизация и лемматизация 

# In[1]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords 
import pymorphy2
from nltk.stem import  WordNetLemmatizer


# In[59]:


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# In[18]:


data = pd.read_csv('annotations.csv')  


# In[29]:


data.drop(data[data['annotation'].isna()].index, inplace=True)


# In[10]:


morph = pymorphy2.MorphAnalyzer()
all_annotations = data.annotation
stop_words = set.union(set(stopwords.words("russian")), set(stopwords.words("english")), set(['это', 'ваш', 'наш', 'свой', 'всё', 'каждый', 'любой', 'весь', 'который']))
pattern = r"(\<(/?[^>]+)>)|[^\w]|\b\d+"
lemmatizer = WordNetLemmatizer()


# In[7]:


def get_tokens(annotation):
    text = re.sub(pattern, " ", annotation)
    tokens = nltk.word_tokenize(str(text))
    lemma = [lemmatizer.lemmatize(morph.parse(token)[0].normal_form) for token in tokens]
    words = [word for word in lemma if not word in stop_words]
    return " ".join(words)


# In[8]:


tokens = data['annotation'].apply(get_tokens)


# In[9]:


new = pd.concat([data['itemid'], tokens], axis=1) 
new.rename(columns={'annotation': 'tokens'}, inplace=True)
new.head()


# In[ ]:


new.to_csv("tokens.csv")


# In[ ]:




