#!/usr/bin/env python
# coding: utf-8

# # Обучение и сохранение модели LDA 

# In[ ]:


import artm
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm_notebook


# In[2]:


batch_vectorizer = artm.BatchVectorizer(data_path='.', data_format='bow_uci',
                                        collection_name='toc', target_folder='ozon_fit')


# In[40]:


lda = artm.LDA(num_topics=100, alpha=0.01, beta=0.001, cache_theta=True,
               num_document_passes=5, dictionary=batch_vectorizer.dictionary)


# In[41]:


lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)


# In[5]:


lda.sparsity_phi_last_value
lda.sparsity_theta_last_value
lda.perplexity_value


# In[10]:


# top_tokens = lda.get_top_tokens(num_tokens=10)
# for i, token_list in enumerate(top_tokens):
#      print('Topic #{0}: {1}'.format(i, token_list))


# In[11]:


phi = lda.phi_ # вероятность термов W в каждой теме t
theta = lda.get_theta() # вероятность тем t в каждом документе D


# In[13]:


theta.to_pickle("theta_100.pkl")


# In[14]:


lda.save('lda_100_p', 'p_wt')


# In[15]:


lda.save('lda_100_n', 'n_wt')


# In[42]:


perplexity = lda.perplexity_value
with open("perplexity_100.txt", "w") as file:
    MyList = map (lambda x: str(x) + ' ', perplexity) 
    file.writelines(MyList) 


# In[ ]:




