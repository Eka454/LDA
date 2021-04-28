#!/usr/bin/env python
# coding: utf-8

# # СОЗДАНИЕ ВХОДНЫХ ДАННЫХ ДЛЯ МОДЕЛИ

# ## Формат входных данных - UCI Bag-of-words  
# 
# Состоит из двух файлов - vocab.*.txt и docword.*.txt 
# 
# ### Формат docword.*.txt файла - 3 строки заголовка, за которыми следуют тройки NNZ:
# 
# D  
# W  
# NNZ  
# docID wordID count  
# docID wordID count  
# docID wordID count  
# docID wordID count  
# 
# ### Формат vocab.*.txt файла - строки со словами  
# 
# word  
# word  
# word  
# 

# In[47]:


import pandas as pd
import collections
from tqdm import tqdm_notebook


# In[48]:


data = pd.read_csv('tokens.csv')
data.drop(data[data['tokens'].isna()].index, inplace=True)


# In[49]:


data.head()


# In[51]:


tokens_set = set() # уникальные слова
N = 0 # общее количество слов
d_t_c = [] # документ => слово => кол-во
for i in tqdm_notebook(data.tokens):
    description = str(i)
    docunent_tokens = description.split(' ')
    cnt_each_token_in_document = collections.Counter()
    for each_token in docunent_tokens:
        cnt_each_token_in_document[each_token] += 1
        N += 1
        tokens_set.add(each_token)
    d_t_c.append(cnt_each_token_in_document)
print(len(tokens_set), N)


# In[52]:


D = len(d_t_c) # количество документов
W = len(tokens_set) # количество уникальных слов


# In[53]:


n_t = dict(zip(tokens_set, [i+1 for i in range(W)])) # слово => номер слова


# In[56]:


with open('docword.toc.txt', 'w', encoding='utf-8') as f:
        d = 0
        f.writelines([str(D), '\n', str(W), '\n', str(N), '\n'])
        for t_c in tqdm_notebook(d_t_c):
            for word, cnt in t_c.items():
                f.writelines([str(d), ' ', str(n_t[word]), ' ', str(cnt), '\n'])
            d += 1


# In[57]:


with open('vocab.toc.txt', 'w', encoding='utf-8') as f:
        for k in tqdm_notebook(n_t.keys()):
                f.writelines([k, '\n'])


# In[ ]:




