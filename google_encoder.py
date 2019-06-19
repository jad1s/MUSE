#!/usr/bin/env python
# coding: utf-8

# In[59]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import tf_sentencepiece


# In[243]:


# Import our sentence file
en_sent = "./an_enzh/train.tok.tc.en"
zh_sent = "./an_enzh/train.tok.tc.zh"

english_sentences = []
chinese_sentences = []
number_sample = 500

import io
with io.open(en_sent, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for i, line in enumerate(f):
        english_sentences.append(line[:-1])
#         print(line);
        if i == number_sample - 1: #sample lines
            break;

print("Loaded %i lines from en_sent." % len(english_sentences))

with io.open(zh_sent, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for i, line in enumerate(f):
        chinese_sentences.append(line[:-1])
#         print(line);
        if i == number_sample - 1: #sample lines
            break;

print("Loaded %i lines from zh_sent." % len(chinese_sentences))


# In[241]:


# filter too long or too short sentences
short_eng_sent = []
short_chi_sent = []
max_char = 150
min_words = 3

print("max_char = %i" % max_char)
print("min_words = %i" % min_words)

for i in range(len(english_sentences)):
    if (len(english_sentences[i]) <= max_char and 
        len(chinese_sentences[i]) <= max_char and
        len(english_sentences[i].split()) >= min_words and 
        len(chinese_sentences[i].split()) >= min_words):
        short_eng_sent.append(english_sentences[i])
        short_chi_sent.append(chinese_sentences[i])

print("Loaded %i lines from short_eng_sent." % len(short_eng_sent))
print("Loaded %i lines from short_chi_sent." % len(short_chi_sent))


# mock up bad translations
import random
mix_short_eng = short_eng_sent
mix_short_chi = short_chi_sent

for i in range(len(short_eng_sent)):
    # random crop from the left, at least min_words
    mix_short_eng.append(" ".join(short_eng_sent[i].split()[0:random.randint(min_words,len(short_eng_sent[i].split()))]))
    mix_short_chi.append(" ".join(short_chi_sent[i].split()[0:random.randint(min_words,len(short_chi_sent[i].split()))]))
#     # random crop & disorder
#     mix_short_eng.append(" ".join(random.sample(uni_short_eng[i].split(), random.randint(1,len(uni_short_eng[i].split())))))
#     mix_short_chi.append(" ".join(random.sample(uni_short_chi[i].split(), random.randint(1,len(uni_short_chi[i].split())))))

print("Loaded %i lines from mix_short_eng." % len(mix_short_eng))
print("Loaded %i lines from mix_short_chi." % len(mix_short_chi))


# filter duplicate entries
eng_chi_dict = {}
for i in range(len(mix_short_eng)):
    eng_chi_dict[mix_short_eng[i]]=mix_short_chi[i]

chi_eng_dict = {v: k for k, v in eng_chi_dict.items()}

uni_short_eng = list(chi_eng_dict.values())
uni_short_chi = list(chi_eng_dict.keys())

print("Loaded %i lines from uni_short_eng." % len(uni_short_eng))
print("Loaded %i lines from uni_short_chi." % len(uni_short_chi))


# The 8-language multilingual module. There are also en-es, en-de, and en-fr bilingual modules.
module_url = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"

# Set up graph.
g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  xling_8_embed = hub.Module(module_url)
  embedded_text = xling_8_embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.Session(graph=g)
session.run(init_op)

# Compute embeddings.
en_result = session.run(embedded_text, feed_dict={text_input: uni_short_eng})
zh_result = session.run(embedded_text, feed_dict={text_input: uni_short_chi})


# In[236]:


# output first line: number of vectors and embedding dimension
# Replace the ‘blank’ with underscore
num_vector = len(uni_short_eng)
emb_dim = len(en_result[0]) 

print("num_vector = %i" % num_vector)
print("emb_dim = %i" % emb_dim)

# output embeddings
with open('./data/src_emb_en.txt', 'w') as f:
    for i in range(len(uni_short_eng)+1):
        if i == 0:
            f.write(str(num_vector)+' '+str(emb_dim)+'\n')
        else:
            f.write(str(uni_short_eng[i-1]).replace(' ','_')+' '+str(en_result[i-1].tolist())[1:-1].replace(',',' ')+'\n')

print("src_emb_en.txt done")

with open('./data/tgt_emb_zh.txt', 'w') as f:
    for i in range(len(uni_short_chi)+1):
        if i == 0:
            f.write(str(num_vector)+' '+str(emb_dim)+'\n')
        else:
            f.write(str(uni_short_chi[i-1]).replace(' ','_')+' '+str(zh_result[i-1].tolist())[1:-1].replace(',',' ')+'\n')

print("tgt_emb_zh.txt done")