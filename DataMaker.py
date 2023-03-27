from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import scipy as sp
import numpy as np
from collections import defaultdict
import pickle as pkl
class Data():
    def __init__(self, data_file , min_freq=1 , max_lengths =40000):
        self.data_file = data_file
        self.freq = defaultdict(int)
        self.max_lengths = max_lengths
        self.min_freq = min_freq
        self.word2idx= self.tokenize_data()
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        #save the word2idx dictionary
        with open('word2idx.pkl', 'wb') as f:
            pkl.dump(self.word2idx, f)
        
    def tokenize_data(self):
        '''This function tokenizes the data file and returns a list of tokens'''
        word2idx = {'<OOV>':0 }
        f = open('tokenised_data.txt', 'w')
        i=0 
        with open(self.data_file, 'r') as f2:
            for line in f2:
                i+=1
                if i== self.max_lengths:
                    break
                tokens = word_tokenize(line)
                tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english')]
                tokens = [token.lower() for token in tokens if token.isalpha()]
                for token in tokens:
                    self.freq[token] += 1
                f.write(str(tokens).replace('\n', ''))
                f.write('\n')
        f.close()
        f1=open('processed_data.txt', 'w')
        new_freq = defaultdict(int)
        with  open('tokenised_data.txt', 'r') as f:
            for line in f:
                tokens = eval(line)
                new_tokens = []
                for token in tokens:
                    if self.freq[token]<= self.min_freq:
                        token = '<OOV>'
                    if token not in word2idx.keys():
                        word2idx.update({token: len(word2idx)})
                    new_tokens.append(word2idx[token])

                f1.write(str(new_tokens).replace('\n', ''))
                f1.write('\n')
        for token in self.freq.keys():
            if self.freq[token] > self.min_freq:
                new_freq[word2idx[token]] += self.freq[token]
            else :
                new_freq[0] += self.freq[token]
        pkl.dump(new_freq, open('freq.pkl', 'wb'))
        f1.close()
        return word2idx

data = Data('less_data.txt', min_freq=1, max_lengths=40000)
            


        