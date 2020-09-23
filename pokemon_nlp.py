
import warnings
warnings.filterwarnings("ignore")

import json
token = {'username':'mohamedshiraz','key':'6c662609a0d765b6ad4b4a12f8063012'}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)

!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json

!kaggle config set -n path -v{/content}

!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets list

!kaggle datasets download -d vishalsubbiah/pokemon-images-and-types -p /content
!unzip '/content/pokemon-images-and-types.zip'

import pandas as pd
df = pd.read_csv("pokemon.csv")

df = df[['Name','Type1']]
df.head()

def wordsplit1(s):
    temp=''
    for i in s:
        if i.isalnum():
            temp += i
        else:
            pass
    s= temp
    vowels = ['a','e','i','o','u','y']
    s=s.lower()
    word = []
    last=0
    n=len(s)
    
    for i in range(0,n):

        if i+1 == n and s[i] in vowels:
            if last == i:
                word[-1]+= str(s[last:i+1])
            else:
                word.append(s[last:])
        if i+1 == n and s[i] not in vowels:
            if last == i:
                word[-1]+= str(s[last:i+1])

        elif s[i] in vowels:
            for j in range(i+1,n):
                if s[j] in vowels:
                    continue
                else:
                    word.append(s[last:(j+1)])
                    last =  j+1
                    break

            
        
    for i in word:
        if i == '' or i == " ":
            try:
                word.remove(i)
                word.remove(i)
            except:
                pass
  
    return " ".join(word)

def wordsplit2(s):
    l=[]
    s = s.lower()
    for i in range (0,len(s),2):
        try:
            l.append(str(s[i]+s[i+1]))
        except:
            l.append(s[i])
    return " ".join(l)
wordsplit2('shiraz')

wordsplit2('shiraz')

df['split'] = df['Name'].apply(lambda x: wordsplit2(x))
df.head()

import numpy as np 
from tensorflow.keras.preprocessing.sequence import pad_sequences
corpus = df['split'].values

import tensorflow_datasets as tfds

tokenizertfds = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus, target_vocab_size=2500)
seq = []
for x in corpus:
    seq.append(tokenizertfds.encode(x))

biggest_seq = max([len(x) for x in seq])
padded2 = np.array(pad_sequences(seq,maxlen = biggest_seq,padding = 'pre'))
xsub,ysub = padded2[:,:-1],padded2[:,-1]
from keras.utils import to_categorical
ysub = to_categorical(ysub)
total_words = len(tokenizertfds.subwords)

#subwords
import tensorflow
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Embedding
from tensorflow.keras.models import Sequential

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Embedding(input_dim = 2500,output_dim = 32,input_length = biggest_seq-1),
    tensorflow.keras.layers.Bidirectional(LSTM(25)),
    tensorflow.keras.layers.Dense(647,activation = 'softmax')])

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(xsub,ysub,epochs=400,verbose =1)

model.save('my_model.h5')

#X--------------------------------------------------X

x = padded2
y = df['Type1']
y = pd.get_dummies(y)
biggest_sequence = max([len(i) for i in x])

model2 = Sequential([
                    Embedding(input_dim = 2500,output_dim = 64,input_length= biggest_sequence),
                    Bidirectional(LSTM(20)),
                    # Dense(64,activation = 'relu'),
                    Dense(18,activation ='softmax')
])
model2.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model2.fit(x,y,epochs = 80,validation_split = 0.2)

def make_name(s):
    res = []
    test = s
    for i in range(2):
        test = wordsplit1(test)
        tokenizedtest = tokenizertfds.encode(test)
        paddedtest = np.array(pad_sequences([tokenizedtest],maxlen = biggest_seq-1,padding = 'pre'))
        pred = model.predict_classes(paddedtest,verbose = 0)
        pred = tokenizertfds.decode(pred)
        if pred == test[-1]:
            continue

        else:
          test = test +" " + pred
          res.append(pred)
    res = [x.rstrip(' ') for x in res]
    return (s  + ''.join(res))

def findtype(s):
    ycols = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
       'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
       'Psychic', 'Rock', 'Steel', 'Water']
    pred =0
    test = s
    test = wordsplit2(test)
    tokenizedtest = tokenizertfds.encode(test)
    paddedtest = np.array(pad_sequences([tokenizedtest],maxlen = biggest_seq,padding = 'pre'))
    pred = int(model2.predict_classes(paddedtest,verbose = 0))
        
    return (ycols[pred])

findtype('vaporeon')

# model2.save('model2.h5')

def pokify(s):
  if s=='soham':
    name = 'somusundaram'
  else:
    name = make_name(s)
  print('Your nickname:',s)
  
  print('Your pokemon name:',name)
  Type = findtype(name)
  print('Your poke-type:',Type)
  
  print('*'*50)

pokify('trump')