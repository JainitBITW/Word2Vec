#importing all libraries
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

class WORD2VEC_SVD():
    '''This class accepts a window size and a data file and returns a word2vec model 
    the model can be used to find the most similar words to a given word. This will create a file of all vectors and their coressponding words in the file named vectors.txt. 
    '''
    
    def __init__(self, window_size, data_file):
        self.window_size = window_size
        self.data_file = data_file
        self.word2idx= pkl.load(open('word2idx.pkl', 'rb'))
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
    
    def calculate_word_vectors(self):
        '''This function calculates the co-occurence matrix and then uses SVD to calculate the word vectors'''
        self.vocab_size = len(self.word2idx)
        print('Calculating word vectors')
        vocab_size = self.vocab_size
        co_occurance_matrix = sp.sparse.lil_matrix((vocab_size, vocab_size ), dtype=np.int8)
        with open('processed_data.txt', 'r') as f:
            for line in f: 
                try :
                    tokens =(eval(line))
                    for center_i, center_word in enumerate(tokens):
                        for w_i in range(-self.window_size, self.window_size + 1):
                            context_i = center_i + w_i
                            if context_i < 0 or context_i >= len(tokens) or center_i == context_i:
                                continue
                            context_word = tokens[context_i]
                            center_idx = center_word
                            context_idx = context_word
                            co_occurance_matrix[center_idx, context_idx] += 1
                except:
                    print(line)
        print('Co-occurence matrix complete')

        U, S, Vt = sp.sparse.linalg.svds(co_occurance_matrix.astype(np.float64), k=200)
        # Reversing the order of the singular values to pick the finest and most contributing dimensions
        U= U[:,::-1]
        S= S[::-1]
        #getting the finest dimensions
        dimensions = 0
        sum_s = sum(S)
        for i in range(len(S)):
            if sum(S[:i])/sum_s > 0.9:
                dimensions = i
                break
        U = U[:, :dimensions]
        S = S[:dimensions]
        U = [u/np.linalg.norm(u) for u in U]
        f= open('vectors.tsv', 'w')
        self.word2vec ={}
        for i in range(len(U)):
            self.word2vec.update({self.idx2word[i]:U[i]})
            tsv_line = '\t'.join([str(x) for x in U[i]])
            f.write(tsv_line)
            f.write('\n')
        f.close()

    def most_similar(self, word):
        '''This function returns the most similar words to the given word'''
        word_vector = self.word2vec[word]
        similarities = []
        for w in self.word2vec.keys():
            if w != word and w != '<OOV>':
                similarities.append((w, np.dot(word_vector, self.word2vec[w])))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:10]

def main():
    w2v = WORD2VEC_SVD(5, 'less_data.txt')
    w2v.calculate_word_vectors()
    f = open('vectors.tsv', 'w')
    f2 = open('metadata.tsv', 'w')
    for word in w2v.word2vec.keys():
        f2.write(word)
        f2.write('\n')
        tsv_line = '\t'.join([str(x) for x in w2v.word2vec[word]])
        f.write(tsv_line)
        f.write('\n')

    print(w2v.most_similar('titanic'))

if __name__ == '__main__':
    main()
