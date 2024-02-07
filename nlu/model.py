import yaml
import numpy as np

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
input_data = np.zeros((len(inputs), max_seq, len(chars)), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k, chr2idx[ch]] = 1
        
print(input_data[4])

'''    
print(inputs)
print(outputs)
'''