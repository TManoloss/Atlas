import yaml
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import to_categorical

data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read()) 

inputs, outputs = [], []


for command in data['commands']:
    inputs.append(command['input'])
    outputs.append('{}\{}'.format(command['entity'], command['action'])) 
    
#Processar texto: palavras, carcteres, bytes, sub-palavras

chars = set()

for input in inputs + outputs:
    for ch in input:
        if ch not in chars:
            chars.add(ch)
            
#mapear char-index

chr2idx = {}
idx2chr = {}

# Iterar sobre os caracteres e seus índices
for i, ch in enumerate(chars):
    chr2idx[ch] = i
    idx2chr[i] = ch

max_seq = max([len(x) for x in inputs])

print('Número de caracteres:', len(chars))
print('Maior sequência:', max_seq)

# Criar dataset one-hot (número de exemplos, tamanho da sequência, número de caracteres)
# Criar dataset disperso (número de exemplos, tamanho da sequência)

# input data one-hot encoding
input_data = np.zeros((len(inputs), max_seq, len(chars)), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k, chr2idx[ch]] = 1

#Input data sparce


        
input_data = np.zeros((len(inputs), max_seq), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k] =chr2idx[ch] = 1
        
#Output data
        
labels = set(outputs)

label2idx = {}
idx2labels = {}
idx = 0  # Initialize idx variable
for k, label in enumerate(labels):  # Use enumerate to get the index
    label2idx[label] = k
    idx2labels[k] = label
    
output_data = []

for output in outputs:
    output_data.append(label2idx[output])

output_data = to_categorical(output_data, len(output_data))
        
print(output_data[0])

model = Sequential()
model.add(Embedding(max_seq, 64))
model.add(LSTM(128, return_sequences=True, unroll=True))
model.add(Dense(len(output_data), activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=['acc'])
model.summary()

model.fit(input_data, output_data, epochs=16)

'''    
print(inputs)
print(outputs)
'''