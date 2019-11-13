'''
Oct 2019
Xinru Yan

This is adapted from XX's work

remember to
import nltk
nltk.download('punkt')
'''
import pickle
import re

import gensim
import nltk
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from nltk.tokenize import regexp_tokenize, sent_tokenize
from torch.utils.data import Dataset
from tqdm import tqdm


def transform(dd, annotated=True):
    dd = re.sub(r"Let\'s", " Let us ", dd)
    dd = re.sub(r"let\'s", " let us ", dd)
    dd = re.sub(r"\'m", " am ", dd)
    dd = re.sub(r"\'ve", " have ", dd)
    dd = re.sub(r"can\'t", " can not ", dd)
    dd = re.sub(r"n\'t", " not ", dd)
    dd = re.sub(r"\'re", " are ", dd)
    dd = re.sub(r"\'d", " would ", dd)
    dd = re.sub(r"\'ll", " will ", dd)
    dd = re.sub(r"y\'all", " you all ", dd)

    return dd


def check_ack_word(word):
    for i in range(0, len(word)):
        if ord(word[i]) < 128:
            pass
        else:
            return 0
    return 1


class Vocab(object):
    def __init__(self, dataset_text=None, dataset_with_annotation=None):

        self.word2id = {}
        self.id2word = {}

        self.pattern = r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

        self.english_punctuations = []
        self.build_vocab(dataset_text, dataset_with_annotation)

        self.vocab_size = len(self.word2id)

    def build_vocab(self, dataset_text=None, dataset_with_annotation=None):
        sentences = []

        words = []

        if dataset_with_annotation is not None:
            raw_data = pd.read_csv(dataset_with_annotation, encoding='utf-8', sep=',', engine='python')
            raw_data = np.array(raw_data)

            np.random.seed(0)
            np.random.shuffle(raw_data)

            for i in tqdm(range(0, raw_data.shape[0])):
                for j in range(1, 45):
                    if raw_data[i][2 * j] is not None:
                        results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                        raw_data[i][2 * j] = np.nan_to_num(raw_data[i][2 * j])
                        if raw_data[i][2 * j] == 0.0:
                            raw_data[i][2 * j] = '0'
                        dd = results.sub(" website ", raw_data[i][2 * j])
                        a = regexp_tokenize(transform(dd), self.pattern)
                        temp = []
                        for k in range(0, len(a)):
                            if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                                if a[k].isdigit():
                                    a[k] = 'number'
                                elif a[k][0] == '$':
                                    a[k] = 'money'
                                elif a[k][-1] == '%':
                                    a[k] = 'percentage'
                                temp.append(a[k].lower())
                                words.append(a[k].lower())

                        if len(temp) > 0:
                            sentences.append(temp)

        if dataset_text is not None:
            raw_data2 = pd.read_csv(dataset_text, encoding='utf-8', sep=',', engine='python')
            raw_data2 = np.array(raw_data2)

            for i in tqdm(range(0, raw_data2.shape[0])):
                try:
                    results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                    dd = results.sub(" website ", raw_data2[i][1])
                    results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                    dd = results.sub(" website ", dd)

                    sent = sent_tokenize(dd)

                    temp_sentences = []

                    for j in range(0, len(sent)):
                        a = regexp_tokenize(transform(sent[j], annotated=False), self.pattern)
                        temp = []
                        for k in range(0, min(300, len(a))):
                            if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                                if a[k].isdigit():
                                    a[k] = 'number'
                                elif a[k][0] == '$':
                                    a[k] = 'money'
                                elif a[k][-1] == '%':
                                    a[k] = 'percentage'
                                temp.append(a[k].lower())
                                words.append(a[k].lower())

                        if len(temp) > 0:
                            temp_sentences.append(temp)

                    for j in range(0, len(temp_sentences)):
                        sentences.append(temp_sentences[j])

                except:
                    pass

        word_frequency = {}
        for i in range(0, len(words)):
            if words[i] in word_frequency:
                word_frequency[words[i]] += 1
            else:
                word_frequency[words[i]] = 1

        self.word2id['<PAD>'] = 0
        self.id2word[0] = '<PAD>'
        self.word2id['<UNK>'] = 1
        self.id2word[1] = '<UNK>'

        self.unk_count = 0
        wid = 2
        for (u, v) in word_frequency.items():
            if v >= 2:
                self.word2id[u] = wid
                self.id2word[wid] = u
                wid += 1
            else:
                self.unk_count += 1


class dataLoaderANN(Dataset):
    #NO_LABEL = 9999

    def __init__(self, vocab, dataset_with_annotation, mode='train', idx=0):
        self.vocab = vocab

        self.max_len = 0
        self.length = []
        self.mode = mode
        self.idx = idx

        self.max_sentence_num = 0

        self.pattern = r"""(?x)                  
                     (?:[A-Z]\.)+          
                     |\$?\d+(?:\.\d+)?%?    
                     |\w+(?:[-']\w+)*      
                     |\.\.\.               
                     |(?:[.,;"'?():-_`])    
                  """

        self.english_punctuations = []
        self.label_set = ['direct-rejection', 'self-pity', 'deflect-responsibility', 'attack-credibility', 'organization-inquiry', 'personal-choice', 'delay-tactic', 'hesitance', 'nitpicking', 'not-a-strategy']
        #self.label_set = ['ERPos+', 'ERPos-', 'ERNeg+', 'ERNeg-',
                          #'EEPos+', 'EEPos-', 'EENeg+', 'EENeg-']
        #assert len(self.label_set) < self.NO_LABEL, 'NO_LABEL must be raised!'

        self.message_id_and_score = {}

        self.message = []
        self.message_id = []
        self.message_sentence_label = []

        self.message_length = []
        self.sentence_length = []

        self.message_all = []

        self.load_data(dataset_with_annotation)

        if self.max_len > 400:
            self.max_len = 400

        for i in range(0, len(self.message)):
            self.message_all.append((self.message_id[i], self.message[i], self.message_sentence_label[i],
                                     self.message_length[i], self.sentence_length[i]))

        self.train = self.message_all[0:int((self.ann_num - 16))]
        self.dev = self.message_all[self.ann_num - 16:self.ann_num - 8]
        self.test = self.message_all[self.ann_num - 8: self.ann_num]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        elif self.mode == 'test':
            return len(self.test)
        elif self.mode == 'dev':
            return len(self.dev)
        elif self.mode == 'test_debug':
            return 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            temp = self.train[idx]
        elif self.mode == 'test':
            temp = self.test[idx]
        elif self.mode == 'dev':
            temp = self.dev[idx]
        elif self.mode == 'test_debug':
            temp = self.test[self.idx]

        message_id = temp[0]
        message = temp[1]
        message_sentence_label = temp[2]
        message_length = temp[3]
        sentence_length = temp[4]

        message_target = self.lookup_score(message_id)

        message_vec = torch.LongTensor(self.message2id(message))

        labels = np.array([11] * self.max_sentence_num)

        for i in range(0, len(message_sentence_label)):
            labels[i] = message_sentence_label[i]
        labels = torch.LongTensor(labels)

        lengths = np.array([0] * self.max_sentence_num)
        for i in range(0, len(sentence_length)):
            lengths[i] = sentence_length[i]

        lengths = torch.LongTensor(lengths)

        # message,  message_id
        return (message_vec, message_target, labels, message_length, lengths)

    def message2id(self, message):
        X = np.zeros([self.max_sentence_num, self.max_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_sentence_num and j < self.max_len:
                    try:
                        id = self.vocab.word2id[si.lower()]
                        X[i][j] = id
                    except:
                        X[i][j] = 1
        return X

    def lookup_label_id(self, s):
        for i in range(0, len(self.label_set)):
            if s == self.label_set[i]:
                return i
        return 10

    def lookup_score(self, id):
        score = self.message_id_and_score[id]
        return score

    def load_data(self, dataset_with_annotation):
        with open('../data/preprocessed/persuasion_donation_label.pkl', 'rb') as f:
            self.message_id_and_score = pickle.load(f)

        raw_data = pd.read_csv(dataset_with_annotation, encoding='utf-8', sep=',', engine='python')
        raw_data = np.array(raw_data)

        np.random.seed(0)
        np.random.shuffle(raw_data)

        for i in tqdm(range(0, raw_data.shape[0])):
            sentence_temp = []
            sentence_label = []
            sentence_length_temp = []
            for j in range(1, 45):
                if raw_data[i][2 * j] is not None:
                    results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                    raw_data[i][2 * j] = np.nan_to_num(raw_data[i][2 * j])
                    if raw_data[i][2 * j] == 0.0:
                        raw_data[i][2 * j] = '0'
                    dd = results.sub(" website ", raw_data[i][2 * j])
                    a = regexp_tokenize(transform(dd), self.pattern)
                    temp = []

                    for k in range(0, len(a)):
                        if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                            if a[k][0] == '$':
                                a[k] = 'money'
                            elif a[k][-1] == '%':
                                a[k] = 'percentage'
                            temp.append(a[k].lower())
                    sentence_temp.append(temp)
                    l = len(temp)
                    sentence_length_temp.append(l)

                    if self.max_len < l:
                        self.max_len = l
                    # print(f'sent {temp}')
                    # print(f'label {raw_data[i][2 * j + 1]}')
                    # print(f'label number {self.lookup_label_id(raw_data[i][2 * j + 1])}')
                    sentence_label.append(self.lookup_label_id(raw_data[i][2 * j + 1]))

            l = len(sentence_temp)
            self.message_length.append(l)
            if self.max_sentence_num < l:
                self.max_sentence_num = l

            self.message.append(sentence_temp)
            self.message_id.append(raw_data[i][0])
            self.message_sentence_label.append(sentence_label)
            self.sentence_length.append(sentence_length_temp)

        self.ann_num = len(self.message)


class dataLoaderUnann(Dataset):
    def __init__(self, vocab, dataset_text):
        self.vocab = vocab

        self.max_len = 900

        self.max_sentence_num = 45

        self.english_punctuations = []

        self.pattern = r"""(?x)                  
                     (?:[A-Z]\.)+          
                     |\$?\d+(?:\.\d+)?%?    
                     |\w+(?:[-']\w+)*      
                     |\.\.\.               
                     |(?:[.,;"'?():-_`])    
                  """

        self.message_id_and_score = {}

        self.message = []
        self.message_id = []

        self.message_length = []
        self.sentence_length = []

        self.message_all = []

        self.load_data(dataset_text)

        for i in range(0, len(self.message)):
            self.message_all.append(
                (self.message_id[i], self.message[i], self.message_length[i], self.sentence_length[i]))

    def __len__(self):
        return len(self.message)

    def __getitem__(self, idx):
        temp = self.message_all[idx]

        message_id = temp[0]
        message = temp[1]
        message_length = temp[2]
        sentence_length = temp[3]

        message_target = self.lookup_score(message_id)

        message_vec = self.message2id(message)

        labels = np.array([11] * self.max_sentence_num)

        lengths = np.array([0] * self.max_sentence_num)

        for i in range(0, min(self.max_sentence_num, len(sentence_length))):
            lengths[i] = sentence_length[i]

        lengths = torch.LongTensor(lengths)

        return (message_vec, message_target, labels, message_length, lengths)

    def message2id(self, message):
        X = np.zeros([self.max_sentence_num, self.max_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_sentence_num and j < self.max_len:
                    try:
                        id = self.vocab.word2id[si.lower()]
                        X[i][j] = id
                    except:
                        X[i][j] = 1

        return X

    def lookup_score(self, id):
        score = self.message_id_and_score[id]
        return score

    def load_data(self, dataset_text):
        with open('../data/preprocessed/persuasion_donation_label.pkl', 'rb') as f:
            self.message_id_and_score = pickle.load(f)

        raw_data = pd.read_csv(dataset_text, encoding='utf-8', sep=',', engine='python')
        raw_data = np.array(raw_data)

        for i in tqdm(range(0, int(raw_data.shape[0]))):
            try:
                sentence_temp = []
                sentence_length_temp = []

                results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" website ", raw_data[i][1])
                results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" website ", dd)

                sent = sent_tokenize(dd)

                for j in range(0, len(sent)):
                    a = regexp_tokenize(transform(sent[j], annotated=False), self.pattern)
                    temp = []
                    for k in range(0, len(a)):
                        if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                            if a[k][0] == '$':
                                a[k] = 'money'
                            elif a[k][-1] == '%':
                                a[k] = 'percentage'
                            temp.append(a[k].lower())

                    if len(temp) > 0:
                        l = min(300, len(temp))
                        sentence_length_temp.append(l)
                        sentence_temp.append(temp[:300])

                if len(sentence_temp) > 0:
                    self.message_id.append(raw_data[i][0])
                    l = len(sentence_temp)

                    self.message_length.append(l)
                    self.sentence_length.append(sentence_length_temp[:45])
                    self.message.append(sentence_temp[:45])

            except:
                pass


if __name__ == '__main__':
    dataset_with_annotation = "../data/preprocessed/annotation_dataset.csv"
    dataset_text = "../data/preprocessed/persuasion_dataset_text.csv"

    vocab = Vocab(dataset_with_annotation=dataset_with_annotation, dataset_text=dataset_text)

    print(f'dataset vcoab size is {vocab.vocab_size}')

    dataSet = dataLoaderANN(vocab, dataset_with_annotation)
    dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode='test')
    dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode='dev')
    dataSet4 = dataLoaderUnann(vocab, dataset_text)

    loaderTrain = Data.DataLoader(dataset=dataSet, batch_size=512, shuffle=False, num_workers=4)
    loaderTest = Data.DataLoader(dataset=dataSet2, batch_size=500, shuffle=False, num_workers=4)
    loaderDev = Data.DataLoader(dataset=dataSet3, batch_size=500, shuffle=False, num_workers=4)
    loaderTrain2 = Data.DataLoader(dataset=dataSet4, batch_size=512, shuffle=False, num_workers=4)

    for step, (x, y, l, num, length) in enumerate(loaderTrain):
        print('train')
        print(x.shape, y.shape, l.shape, num.shape, length.shape)

    for step, (x, y, l, num, length) in enumerate(loaderTrain2):
        print('train2')
        print(x.shape, y.shape, l.shape, num.shape, length.shape)

    for step, (x, y, l, num, length) in enumerate(loaderTest):
        print('test')
        print(x.shape, y.shape, l.shape, num.shape, length.shape)

    for step, (x, y, l, num, length) in enumerate(loaderDev):
        print('dev')
        print(x.shape, y.shape, l.shape, num.shape, length.shape)