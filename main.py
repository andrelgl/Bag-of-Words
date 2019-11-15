import os
import re
import pandas as pd
import numpy as np

def extract(sentence):
    ignore = ['a', 'an', 'the']
    words = re.sub("[^\w]", " ", sentence)
    toLowerSentences = [frag.lower() for frag in words if frag not in ignore]
    return toLowerSentences

def tokenize(sentence):
    wordSentence = extract(sentence)
    return sorted(list(wordSentence))

def bagofwords(sentence, words):
    sentence = extract(sentence)
    bag = np.zeros(len(words))
    for sw in sentence:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1
    return np.array(bag)

def cosine(v, w):
    return np.dot(v, w) / np.sqrt(np.dot(v, v) * np.dot(w, w))

def transformList(list):
    string = ''
    for i in list:
        string += ', '.join(i)
    return string

def bagofwordsData(csvFile, words):
    sentenceWords = extract(transformList(csvFile))
    bag = np.zeros(len(words))
    for string in sentenceWords:
        for i, word in enumerate(words):
            if word == string:
                bag[i] += 1

    return np.array(bag)

queryInput = input("Busca: ")

query = bagofwords(queryInput, tokenize(queryInput))
bagofwords = []

for filename in os.listdir('files'):
    csv = pd.read_csv('files/' + filename, sep=";")
    tokens = tokenize(queryInput)
    bagofwords.append((filename, bagofwordsData(csv.values, tokens)))

result = []
for file, bag in bagofwords:
    result.append((file, cosine(query, bag)))

result.sort(key=lambda x: x[1], reverse=True)

print('Similaridade: \n')
for name, sim in result:
    print(name + " : " + str(sim))

