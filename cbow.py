import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pickle as pkl
from scipy.spatial.distance import cosine
# import cuda 
import torch.cuda as cuda


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        input_embeds = self.embeddings(inputs)
        embeds = torch.mean(input_embeds, dim=1)
        out = self.linear(embeds)
        return F.log_softmax(out, dim=1)


class Word2Vec: 
    def __init__(self,data_file, word2idx, context_size=2,embedding_size=50, oov_threshold=2, neg_sample_size=5, lr=0.5):
        self.data_file = data_file
        self.word2idx = word2idx
        self.freq = pkl.load(open('freq.pkl', 'rb'))
        self.freq_dist = np.array(list(self.freq.values()))
        self.vocabulary = list(self.word2idx.keys())
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.oov_threshold = oov_threshold
        self.neg_sample_size = neg_sample_size
        self.lr = lr
        self.BATCH_SIZE = 64
        self.oov_token = '<OOV>'
        self.vocab_size = len(self.word2idx)
        self.model = CBOW(self.vocab_size, self.embedding_size)
        self.weights = self.negative_sampling()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = nn.NLLLoss()
        self.dataset = self.create_dataset()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
    def create_dataset(self):
        print("Creating Dataset")
        dataset = []
        with open(self.data_file, 'r') as f:
            for line in f:
                tokens = eval(line)

                for i in range(self.context_size, len(tokens) - self.context_size):
                    focus_index = tokens[i]
                    context_indices = []
                    for j in range(i - self.context_size, i + self.context_size + 1):
                        if i == j:
                            continue
                        context_index = tokens[j]
                        context_indices.append(context_index)
                    dataset.append((context_indices, focus_index))
        return dataset
    
    def negative_sampling(self):
        normalized_freq = F.normalize(
            torch.Tensor(self.freq_dist).pow(0.75), dim=0)
        weights = torch.ones(len(self.freq_dist)).cuda()

        for _ in range(len(self.freq_dist)):
            for _ in range(self.neg_sample_size):
                neg_index = torch.multinomial(normalized_freq, 1)[0]
                weights[neg_index] += 1

        return weights
    
    def train(self, num_epochs):
        self.model.to(self.device)
        print(self.device)
        losses = []
        loss_fn = nn.NLLLoss(weight=self.weights)
        for epoch in range(num_epochs):
            if epoch % 2 == 0 and epoch != 0 and self.optimizer.param_groups[0]['lr'] > 0.001:
                self.optimizer.param_groups[0]['lr'] /= 2
                print(f"changed Learning Rate: {self.optimizer.param_groups[0]['lr']}")
            print(f"Epoch {epoch}")
            net_loss = 0
            for i in range(0, len(self.dataset), self.BATCH_SIZE):
                batch = self.dataset[i: i + self.BATCH_SIZE]

                context = [x[0] for x in batch]
                focus = [x[1] for x in batch]

                context_var = Variable(torch.cuda.LongTensor(context))
                focus_var = Variable(torch.cuda.LongTensor(focus))
                context_var = context_var.to(self.device)
                focus_var = focus_var.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(context_var)
                loss = loss_fn(log_probs, focus_var)
                loss.backward()
                self.optimizer.step()

                net_loss += loss.item()
            print(f"Loss: {loss.item()}")
            losses.append(net_loss)

    def get_embedding(self , word_idx):
        embedding_index = Variable(torch.cuda.LongTensor([word_idx]))
        return self.model.embeddings(embedding_index).data[0]
    
    def get_similar_words(self, word, topn=10):
        word_idx = self.word2idx[word]
        word_embedding = self.get_embedding(word_idx)
        similarities = []
        for i in self.word2idx.values():
            if i == word_idx:
                continue
            sim = cosine(word_embedding, self.get_embedding(i))
            similarities.append((self.vocabulary[i], sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

word2idx = pkl.load(open('word2idx.pkl', 'rb'))
encoder = Word2Vec('processed_data.txt', word2idx, context_size=2, embedding_size=50, oov_threshold=2, neg_sample_size=5, lr=0.5)
encoder.train(10)
