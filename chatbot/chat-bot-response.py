#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# importando bibliotecas
import numpy as np
import tflearn
import tensorflow as tf
import random


# In[2]:


# carregando a estrutura da rede
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# carregando as intencoes
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)


# In[3]:


# Construindo a rede
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Deinindo configuracoes do Tensorboard - proxima postagem ....
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


# In[4]:


def clean_up_sentence(sentence):
    # tokenizando frases
    sentence_words = nltk.word_tokenize(sentence)
    # stem 
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# In[31]:


p = bow("O que vocês vendem ?", words)
print (p)
print (classes)


# In[6]:


# load our saved model
model.load('./model.tflearn')


# In[7]:


#criando contexto do usuario 
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                   
                    if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # sorteia uma resposta da intenção
                        return print("ChatBot:",random.choice(i['responses']))

            results.pop(0)

#funcao para receber frases e dar respostas com detalhes
def mensagem(frase):
    response(frase, show_details=True)

#example
mensagem("oi tudo bem ?")




