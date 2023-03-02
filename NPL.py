#!/usr/bin/env python
# coding: utf-8

# In[63]:


import spacy_streamlit
models = ["en_core_web_sm", "en_core_web_md"]


# In[64]:


nlp=spacy.load("en_core_web_sm")


# In[65]:


my_text = """show me month to date overallscrap"""


# In[66]:


my_doc = nlp(my_text)


# In[67]:


nlp.pipe_names


# In[68]:


for token in my_doc:
  print(token.text,'--',token.is_stop,'---',token.is_punct)


# In[69]:


my_doc_cleaned = [token for token in my_doc if not token.is_stop and not token.is_punct]

for token in my_doc_cleaned:
  print(token.text)


# In[70]:


for token in my_doc:
  print(token.lemma_)


# In[71]:


for token in my_doc:
    print(token.text, "-->", token.dep_)


# In[72]:


import numpy as np
text1 = 'Cnt'
text2 = 'Count'
doc1 = nlp(text1)
doc2 = nlp(text2)
print("spaCy :", doc1.similarity(doc2))

print(np.dot(doc1.vector, doc2.vector) / (np.linalg.norm(doc1.vector) * np.linalg.norm(doc2.vector)))


# In[ ]:
