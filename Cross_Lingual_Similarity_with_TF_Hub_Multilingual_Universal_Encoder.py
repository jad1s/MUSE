#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Install the latest TensorFlow version compatible with tf-sentencepiece.
# !pip3 install --quiet tensorflow==1.12.0
# # Install TF-Hub.
# !pip3 install --quiet tensorflow-hub
# !pip3 install --quiet seaborn
# # Install Sentencepiece.
# !pip3 install --quiet tf-sentencepiece


# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import tf_sentencepiece


# In[2]:


import io

def load_sent(filepath, number_sample):
    sentences = []
    with io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            sentences.append(line[:-1])
            if i == number_sample - 1: #sample lines
                break;
    print("Loaded %i lines " % len(sentences) + "from %s." % filepath)
    return sentences


# In[3]:


# Import our sentence file
en_sent = "./an_enzh/train.tok.tc.en"
zh_sent = "./an_enzh/train.tok.tc.zh"

number_sample = 20

english_sentences = load_sent(en_sent, number_sample)
chinese_sentences = load_sent(en_sent, number_sample)


# In[4]:


# filter short sentences
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
print("Loaded %i lines from short_chi_sent." % len(short_chi_sent)+'\n')


# In[5]:


# filter long sentences
long_eng_sent = []
long_chi_sent = []
min_char = max_char

print("min_char = %i" % min_char)

for i in range(len(english_sentences)):
    if (len(english_sentences[i]) > min_char or 
        len(chinese_sentences[i]) > min_char):
        long_eng_sent.append(english_sentences[i])
        long_chi_sent.append(chinese_sentences[i])

print("Loaded %i lines from long_eng_sent." % len(long_eng_sent))
print("Loaded %i lines from long_chi_sent." % len(long_chi_sent)+'\n')


# In[6]:


import random

def under_tran(sentence, times):
    sentence = sentence.split()
    times = min(times, len(sentence)-1) # avoid out of index
    for i in range(times):
        index = random.randint(0, len(sentence)-1)
        sentence.remove(sentence[index])
    return sentence

def over_tran(sentence, times):
    sentence = sentence.split()
    for i in range(times):
        index = random.randint(0, len(sentence)-1)
        sentence.insert(index, sentence[index])
    return sentence

def mock_tran(eng_sent, chi_sent):
    mix_eng = eng_sent
    mix_chi = chi_sent

    num_origins = len(eng_sent)

    # undertranslation: unnecessary ommisions (words not translated)
    for i in range(num_origins):
        # randomly remove 3 words from the sentence
        mix_eng.append(" ".join(under_tran(eng_sent[i], 3)))
        mix_chi.append(" ".join(under_tran(chi_sent[i], 3)))

    print("Loaded %i lines, including undertranslation." % len(mix_eng))
    print("Loaded %i lines, including undertranslation." % len(mix_chi)+'\n')

    # overtranslation: unnecessary addtions 
    for i in range(num_origins):
        # add 3 duplicate words into the sentence
        mix_eng.append(" ".join(over_tran(eng_sent[i], 3)))
        mix_chi.append(" ".join(over_tran(chi_sent[i], 3)))

    print("Loaded %i lines, including undertranslation & overtranslation." % len(mix_eng))
    print("Loaded %i lines, including undertranslation & overtranslation." % len(mix_chi)+'\n')

    # mistranslation: low accuracy, fluency, punctuation (disordered sentence)
    for i in range(num_origins):
        # random disorder # random.shuffle didn't work (?)
        mix_eng.append(" ".join(random.sample(eng_sent[i].split(), len(eng_sent[i].split()))))
        mix_chi.append(" ".join(random.sample(chi_sent[i].split(), len(chi_sent[i].split()))))

    print("Loaded %i lines, including undertranslation, overtranslation & mistranslation." % len(mix_eng))
    print("Loaded %i lines, including undertranslation, overtranslation & mistranslation." % len(mix_chi)+'\n')
    
    return mix_eng, mix_chi


# In[7]:


# mock up bad translations for short sentences
print('\n'+"<---mock up bad translations for short sentences--->"+'\n')
mix_short_eng, mix_short_chi = mock_tran(short_eng_sent, short_chi_sent)

# mock up bad translations for long sentences
print('\n'+"<---mock up bad translations for long sentences--->"+'\n')
mix_long_eng, mix_long_chi = mock_tran(long_eng_sent, long_chi_sent)


# In[8]:


def write_file(filepath, file):
    with open(filepath, 'w') as f:
        for i in range(len(file)):
            f.write(str(file[i])+'\n')
    print("%s done" % filepath)


# In[9]:


write_file('./data/mix_short_eng', mix_short_eng)
write_file('./data/mix_short_chi', mix_short_chi)
write_file('./data/mix_long_eng', mix_long_eng)
write_file('./data/mix_long_chi', mix_long_chi)


# In[10]:


def filter_dup(eng, chi):
    eng_chi_dict = {}
    for i in range(len(eng)):
        eng_chi_dict[eng[i]]=chi[i]
        
    chi_eng_dict = {v: k for k, v in eng_chi_dict.items()}

    uni_eng = list(chi_eng_dict.values())
    uni_chi = list(chi_eng_dict.keys())
    
    print("Loaded %i lines " % len(uni_eng))
    print("Loaded %i lines " % len(uni_chi))
    return uni_eng, uni_chi


# In[11]:


print('\n'+"<---filter duplicate entries for short sentences--->"+'\n')
uni_short_eng, uni_short_chi = filter_dup(mix_short_eng, mix_short_chi)

print('\n'+ "<---filter duplicate entries for long sentences--->"+'\n')
uni_long_eng, uni_long_chi = filter_dup(mix_long_eng, mix_long_chi)


# In[12]:


def encode_tran(eng, chi):
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
    en_result = session.run(embedded_text, feed_dict={text_input: eng})
    zh_result = session.run(embedded_text, feed_dict={text_input: chi})
    
    return en_result, zh_result


# In[13]:


print('\n'+"<---encode short sentences--->"+'\n')
short_en_emb, short_zh_emb = encode_tran(uni_short_eng, uni_short_chi)

print('\n'+ "<---encode long sentences--->"+'\n')
long_en_emb, long_zh_emb = encode_tran(uni_long_eng, uni_long_chi)


# In[14]:


# output first line: number of vectors and embedding dimension
# Replace the ‘blank’ with underscore

def write_emb(filepath, sentence, emb_result):
    num_vector = len(sentence)
    emb_dim = len(emb_result[0]) 

    print("num_vector = %i" % num_vector)
    print("emb_dim = %i" % emb_dim)

    # output embeddings
    with open(filepath, 'w') as f:
        for i in range(len(sentence)+1):
            if i == 0:
                f.write(str(num_vector)+' '+str(emb_dim)+'\n')
            else:
                f.write(str(sentence[i-1]).replace(' ','_')+' '+str(emb_result[i-1].tolist())[1:-1].replace(',',' ')+'\n')

    print("%s done" % filepath)


# In[15]:


print('\n'+"<---export embeddings for short sentences--->"+'\n')
write_emb('./data/src_emb_en_s.txt', uni_short_eng, short_en_emb)
write_emb('./data/src_emb_zh_s.txt', uni_short_chi, short_zh_emb)

print('\n'+"<---export embeddings for long sentences--->"+'\n')
write_emb('./data/src_emb_en_l.txt', uni_long_eng, long_en_emb)
write_emb('./data/src_emb_zh_l.txt', uni_long_chi, long_zh_emb)


# In[215]:


from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
from src.logger import create_logger
from src.dictionary import Dictionary

from logging import getLogger
logger = getLogger()


# In[240]:


_emb_dim_file = 512 # default=300, help="Embedding dimension"
max_vocab = 200000 # default=200000, help="Maximum vocabulary size (-1 to disable)"
full_vocab=False
"""
Reload pretrained embeddings.
- `full_vocab == False` means that we load the `params.max_vocab` most frequent words.
  It is used at the beginning of the experiment.
  In that setting, if two words with a different casing occur, we lowercase both, and
  only consider the most frequent one. For instance, if "London" and "london" are in
  the embeddings file, we only consider the most frequent one, (in that case, probably
  London). This is done to deal with the lowercased dictionaries.
- `full_vocab == True` means that we load the entire embedding text file,
  before we export the embeddings at the end of the experiment.
"""

word2id = {}
vectors = []
   
# def read_txt_embeddings(params, source, full_vocab):
# with io.open(wiki_en_sent, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
with io.open(src_emb_en_s, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for i, line in enumerate(f):
        if i == 0:
            split = line.split()
            assert len(split) == 2 # first line contains two numbers: ??? and embedding dimension
            assert _emb_dim_file == int(split[1]) # second number is num_dim
        else:
            word, vect = line.rstrip().split(' ', 1) #get the word & vectors
#             print('else-word:'+word)
#             print('else-vect:'+vect[0])
#             print('type(vect):'+str(type(vect)))
            if not full_vocab: # will lower case and read the most frequent words
                word = word.lower()
#                 print('if not full_vocab:'+word)
            vect = np.fromstring(vect, sep=' ')
#             print('np.fromstring:'+str(vect.tolist()))
#             print('np.linalg.norm(vect)='+str(np.linalg.norm(vect).tolist()))
            if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                vect[0] = 0.01 # IndexError: index 0 is out of bounds for axis 0 with size 0
            if word in word2id:
                if full_vocab:
                    pass
#                     logger.warning("Word '%s' found twice in %s embedding file"
#                                    % (word, 'source' if source else 'target'))
            else:
                if not vect.shape == (_emb_dim_file,):
                    pass
#                     logger.warning("Invalid dimension (%i) for %s word '%s' in line %i."
#                                    % (vect.shape[0], 'source' if source else 'target', word, i))
                    continue
                assert vect.shape == (_emb_dim_file,), i
                word2id[word] = len(word2id)
#                 print(vect[None]+'\n')
                vectors.append(vect[None])
        if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
            break

assert len(word2id) == len(vectors)
logger.info("Loaded %i pre-trained word embeddings." % len(vectors))

# compute new vocabulary / embeddings
id2word = {v: k for k, v in word2id.items()}
# dico = Dictionary(id2word, word2id, lang)


# In[233]:


embeddings = np.concatenate(vectors, 0)
# embeddings = torch.from_numpy(embeddings).float()
# embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings

# assert embeddings.size() == (len(dico), params.emb_dim)
# return dico, embeddings


# In[10]:


# Import wiki sentence file to test FB scripts
wiki_en_sent = "./wiki.en.vec"
wiki_zh_sent = "./wiki.zh.vec"

wiki_en_sentences = []
wiki_zh_sentences = []
number_sample = 100

import io
with io.open(wiki_en_sent, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for i, line in enumerate(f):
        wiki_en_sentences.append(line[:-1])
#         print(line);
        if i == number_sample - 1: #sample lines
            break;
            
print("Loaded %i lines from wiki_en_sentences." % len(wiki_en_sentences))

with io.open(wiki_zh_sent, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for i, line in enumerate(f):
        wiki_zh_sentences.append(line[:-1])
#         print(line);
        if i == number_sample - 1: #sample lines
            break;
        
print("Loaded %i lines from wiki_zh_sentences." % len(wiki_zh_sentences))


# In[2]:


# output sampled wiki sent embeddings
with open('./data/wiki_en_sentences.txt', 'w') as f:
    for i in range(len(wiki_en_sentences)):
        f.write(str(wiki_en_sentences[i])+'\n')

print("wiki_en_sentences.txt done")

with open('./data/wiki_zh_sentences.txt', 'w') as f:
    for i in range(len(wiki_zh_sentences)):
        f.write(str(wiki_zh_sentences[i])+'\n')

print("wiki_zh_sentences.txt done")


# In[144]:


nmax = 50000

en_word2id = {}
en_id2word = {}
for i in range(len(uni_short_eng2)):
    assert i not in en_word2id, 'word found twice'
    en_word2id[uni_short_eng2[i]] = len(en_word2id) #problem: what if sentences are duplicate?
#     print(en_word2id[uni_short_eng2[i]])
    if len(en_word2id) == nmax:
        break
en_id2word = {v: k for k, v in en_word2id.items()}

zh_word2id = {}
zh_id2word = {}
for i in range(len(uni_short_chi2)):
    assert i not in zh_word2id, 'word found twice'
    zh_word2id[uni_short_chi2[i]] = len(zh_word2id) #problem: what if sentences are duplicate?
#     print(zh_word2id[uni_short_chi2[i]])
    if len(zh_word2id) == nmax:
        break
zh_id2word = {v: k for k, v in zh_word2id.items()}


# In[147]:


def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        
# printing nearest neighbors in the source space
# get_nn(english_sentences[0], en_result, en_id2word, en_result, en_id2word, K=5)
get_nn(uni_short_chi2[2], zh_result, zh_id2word, zh_result, zh_id2word, K=5)


# ## Visualize Embedding Similarity
# With the sentence embeddings now in hand, we can visualize semantic similarity across different languages.

# In[25]:


def visualize_similarity(embeddings_1, embeddings_2, labels_1, labels_2, plot_title):
  corr = np.inner(embeddings_1, embeddings_2)
  g = sns.heatmap(corr,
                  xticklabels=labels_1,
                  yticklabels=labels_2,
                  vmin=0,
                  vmax=1,
                  cmap="YlOrRd")
  g.set_yticklabels(g.get_yticklabels(), rotation=0)
  g.set_title(plot_title)


# ### English-Chinese Similarity

# In[26]:


visualize_similarity(en_result, zh_result, english_sentences, chinese_sentences, "English-Italian Similarity")

