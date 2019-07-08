import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import tf_sentencepiece


# In[26]:


import datetime
import io

def load_sent(filepath, number_sample):
    sentences = []
    with io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            sentences.append(line[:-1])
            if i == number_sample - 1: #sample lines
                break;
    print(str(datetime.datetime.now()) + ": Loaded %i lines " % len(sentences) + "from %s." % filepath)
    return sentences


# In[27]:


# Import our sentence file
en_sent = "./an_enzh/train.tok.tc.en"
zh_sent = "./an_enzh/train.tok.tc.zh"

number_sample = 1000

english_sentences = load_sent(en_sent, number_sample)
chinese_sentences = load_sent(zh_sent, number_sample)


# In[28]:


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

print(str(datetime.datetime.now()) + ": Loaded %i lines from short_eng_sent." % len(short_eng_sent))
print(str(datetime.datetime.now()) + ": Loaded %i lines from short_chi_sent." % len(short_chi_sent)+'\n')


# In[29]:


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

print(str(datetime.datetime.now()) + ": Loaded %i lines from long_eng_sent." % len(long_eng_sent))
print(str(datetime.datetime.now()) + ": Loaded %i lines from long_chi_sent." % len(long_chi_sent)+'\n')


# In[30]:


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

    print(str(datetime.datetime.now()) + 
          ": Loaded %i lines, including undertranslation." % len(mix_eng))
    print(str(datetime.datetime.now()) + 
          ": Loaded %i lines, including undertranslation." % len(mix_chi)+'\n')

    # overtranslation: unnecessary addtions 
    for i in range(num_origins):
        # add 3 duplicate words into the sentence
        mix_eng.append(" ".join(over_tran(eng_sent[i], 3)))
        mix_chi.append(" ".join(over_tran(chi_sent[i], 3)))

    print(str(datetime.datetime.now()) + ": Loaded %i lines, including undertranslation & overtranslation." % len(mix_eng))
    print(str(datetime.datetime.now()) + ": Loaded %i lines, including undertranslation & overtranslation." % len(mix_chi)+'\n')

    # mistranslation: low accuracy, fluency, punctuation (disordered sentence)
    for i in range(num_origins):
        # random disorder # random.shuffle didn't work (?)
        mix_eng.append(" ".join(random.sample(eng_sent[i].split(), len(eng_sent[i].split()))))
        mix_chi.append(" ".join(random.sample(chi_sent[i].split(), len(chi_sent[i].split()))))

    print(str(datetime.datetime.now()) + ": Loaded %i lines, including undertranslation, overtranslation & mistranslation." % len(mix_eng))
    print(str(datetime.datetime.now()) + ": Loaded %i lines, including undertranslation, overtranslation & mistranslation." % len(mix_chi)+'\n')
    
    return mix_eng, mix_chi


# In[31]:


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
            f.write(str(file[i]).replace(' ','_')+'\n')
    print(str(datetime.datetime.now()) + ": %s done" % filepath)


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
    
    print(str(datetime.datetime.now()) + ": Loaded %i lines " % len(uni_eng))
    print(str(datetime.datetime.now()) + ": Loaded %i lines " % len(uni_chi))
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
    print(str(datetime.datetime.now()) + ": Loaded %i lines " % len(en_result))
    
    zh_result = session.run(embedded_text, feed_dict={text_input: chi})
    print(str(datetime.datetime.now()) + ": Loaded %i lines " % len(zh_result))
    
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

    print(str(datetime.datetime.now()) + ": num_vector = %i" % num_vector)
    print(str(datetime.datetime.now()) + ": emb_dim = %i" % emb_dim)

    # output embeddings
    with open(filepath, 'w') as f:
        for i in range(len(sentence)+1):
            if i == 0:
                f.write(str(num_vector)+' '+str(emb_dim)+'\n')
            else:
                f.write(str(sentence[i-1]).replace(' ','_')+' '+str(emb_result[i-1].tolist())[1:-1].replace(',',' ')+'\n')

    print(str(datetime.datetime.now()) + ": %s done" % filepath)


# In[15]:


print('\n'+"<---export embeddings for short sentences--->"+'\n')
write_emb('./data/src_emb_en_s.txt', uni_short_eng, short_en_emb)
write_emb('./data/tgt_emb_zh_s.txt', uni_short_chi, short_zh_emb)

print('\n'+"<---export embeddings for long sentences--->"+'\n')
write_emb('./data/src_emb_en_l.txt', uni_long_eng, long_en_emb)
write_emb('./data/tgt_emb_zh_l.txt', uni_long_chi, long_zh_emb)