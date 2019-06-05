import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import *

from Config import Config
from dataLoader import Vocab, dataLoaderANN, dataLoaderUnann
from AttnModel import Request
from MyLoss import MessageLoss

config = Config()

dataset_text = "./dataSetNew/persuasion_dataset_text.0720.public.csv"
dataset_with_annotation = "./dataSetNew/annotation_dataset.0720.public.csv"


vocab = Vocab(dataset_with_annotation = dataset_with_annotation, dataset_text = dataset_text)
#vocab = Vocab(dataset_with_annotation = dataset_with_annotation, dataset_text = None)

#with open('./data/vocab_new.pkl', 'rb') as f:
#    vocab = pickle.load(f)
print(vocab.vocab_size)


dataSet = dataLoaderANN(vocab, dataset_with_annotation)
dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode = 'test')
dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode = 'dev')
dataSet4 = dataLoaderUnann(vocab, dataset_text)

loaderTrainAnn = Data.DataLoader(dataset = dataSet, batch_size = 128, shuffle = True, num_workers = 0)
loaderTrainUnann = Data.DataLoader(dataset = dataSet4, batch_size = 128, shuffle = True, num_workers = 0)
#loaderTest = Data.DataLoader(dataset = dataSet2, batch_size = 500, shuffle = False, num_workers = 1)
loaderDev = Data.DataLoader(dataset = dataSet3, batch_size = 200, shuffle = False, num_workers = 1)

#request = Request(config, vocab_size = vocab.vocab_size, pretrained_embedding= vocab.embed)
request = Request(config, vocab_size = vocab.vocab_size)

request.cuda()

messageLoss = MessageLoss(w2 = 10)
messageLoss.cuda()

learning_rate = 5e-5
optimizer = torch.optim.Adam(params = request.parameters(), lr = learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.9)


for name, param in request.named_parameters():
    print(name, param.size(), param.requires_grad)
    

def testModel():
    global request
    request.eval()
    for step, (x, y, l, num, length) in enumerate(loaderTest):
        message_input = Variable(x.type(torch.LongTensor)).cuda()
        message_target = Variable(y.type(torch.FloatTensor)).cuda()
        sentence_label = Variable(l.type(torch.LongTensor)).cuda()
     
        sentence_out, message_out = request(message_input, num, length)

        loss, rmse, sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r = messageLoss(labeled_doc = message_out, target1 = message_target, labeled_sent = sentence_out, target2 = sentence_label, mode = 'test')

    if p + r != 0:
        f1 = (2*p*r)/(p+r)
    else:
        f1 = 0

    print("...")
    print("Test: total_loss: {0}, score_rmse_loss: {1}, cross_loss: {2}".format(loss, rmse, sent_loss))
    print("   : corrext: {0}, count : {1}, acc: {2}".format(correct, count, correct/count))
    print("   : acc: {0}, P: {1}, R: {2}, F1: {3}".format(correct/count, p, r, f1))
    print("...")

def evaluation():
    global request
    request.eval()
    
    correct_ = 0
    count_ = 0
    
    for step, (x, y, l, num, length) in enumerate(loaderDev):
        message_input = Variable(x.type(torch.LongTensor)).cuda()
        message_target = Variable(y.type(torch.FloatTensor)).cuda()
        sentence_label = Variable(l.type(torch.LongTensor)).cuda()
     
        sentence_out, message_out = request(message_input, num, length)

        loss, rmse, sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r = messageLoss(labeled_doc = message_out, target1 = message_target, labeled_sent = sentence_out, target2 = sentence_label, mode = 'dev')
        correct_ += correct
        count_ += count

    if p + r != 0:
        f1 = (2*p*r)/(p+r)
    else:
        f1 = 0

    print("...")
    print("Dev: total_loss: {0}, score_rmse_loss: {1}, cross_loss: {2}".format(loss, rmse, sent_loss))
    print("   : corrext: {0}, count : {1}, acc: {2}".format(correct_, count_, correct_/count_))
    #print("   : acc: {0}, P: {1}, R: {2}, F1: {3}".format(correct/count, p, r, f1))
    print("...")

    return correct_/count_, loss, f1


def getAnnTrainBatches():
    ann_data = []
    ann_num = -1
    
    for step, (x, y, l, num, length) in enumerate(loaderTrainAnn):
        ann_data.append((x,y,l,num,length))
        ann_num += 1

    return ann_num, ann_data

def getUnannTrainBatches():
    unann_data = []
    unann_num = -1
        
    for step, (x, y, l, num, length) in enumerate(loaderTrainUnann):
        unann_data.append((x,y,l,num,length))
        unann_num += 1

    return unann_num, unann_data

def trainStep(without_unann = False):
    #global learning_rate
    #global optimizer
    global request

    max_acc = 0
    
    min_loss = 1000

    ann_num = -1
    unann_num = -1

    step = 0
    flag = 0

    while step < 200:
        if ann_num == -1:
            ann_num, ann_data = getAnnTrainBatches()

            acc, loss, f1 = evaluation()

            #scheduler.step()
            #if loss < min_loss:
            if acc >= max_acc or loss <= min_loss:
                if acc >= max_acc: 
                #and loss <= min_loss:
                    print('update!!!')
                    
                    flag = 1
                    torch.save(request, './model/model_attn_1.pkl')
                max_acc = max(acc, max_acc)
                min_loss = min(loss, min_loss)
                

        if unann_num == -1:
            unann_num, unann_data = getUnannTrainBatches()
            step +=1
            if flag >= 10:
                print('early stop!!!')
                break
            else:
                flag += 1
            #scheduler.step()
            

        request.train()
        
        if unann_num%1==0 or without_unann:
            ann_message_input = Variable(ann_data[ann_num][0].type(torch.LongTensor)).cuda()
            ann_message_target = Variable(ann_data[ann_num][1].type(torch.FloatTensor)).cuda()
            ann_sentence_label = Variable(ann_data[ann_num][2].type(torch.LongTensor)).cuda()

            ann_sentence_out, ann_message_out = request(ann_message_input, ann_data[ann_num][3], ann_data[ann_num][4])

            ann_num = ann_num - 1
        #print(len(unann_data), unann_num)
        else:
            ann_message_input = Variable(ann_data[ann_num][0].type(torch.LongTensor)).cuda()
            ann_message_target = Variable(ann_data[ann_num][1].type(torch.FloatTensor)).cuda()
            ann_sentence_label = Variable(ann_data[ann_num][2].type(torch.LongTensor)).cuda()

            ann_sentence_out, ann_message_out = request(ann_message_input, ann_data[ann_num][3], ann_data[ann_num][4])
            ann_sentence_label = None
            
            

        if not without_unann:
            unann_message_input = Variable(unann_data[unann_num][0].type(torch.LongTensor)).cuda()
            unann_message_target = Variable(unann_data[unann_num][1].type(torch.FloatTensor)).cuda()
            unann_sentence_label = Variable(unann_data[unann_num][2].type(torch.LongTensor)).cuda()
            unann_sentence_out, unann_message_out = request(unann_message_input, unann_data[unann_num][3], unann_data[unann_num][4])
            
            if unann_num%1==0:
                w1 = ann_message_input.shape[0]/(ann_message_input.shape[0] + unann_message_input.shape[0])
            else:
                w1 = 0
            

        unann_num = unann_num - 1


        if without_unann:
            loss, labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r = messageLoss(labeled_doc = ann_message_out, target1 = ann_message_target, labeled_sent = ann_sentence_out, target2 = ann_sentence_label, w1 = 0, unlabeled_doc = None, target3 = None, mode = 'train')
            
        else:
            loss, labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r = messageLoss(labeled_doc = ann_message_out, target1 = ann_message_target, labeled_sent = ann_sentence_out, target2 = ann_sentence_label, w1 = w1, unlabeled_doc = unann_message_out, target3 = unann_message_target, mode = 'train')
            



        optimizer.zero_grad()
        loss.backward()
        #labeled_sent_loss.backward()
        optimizer.step()

        print("Train Step {0}: loss: {1}, labeled_sent_loss: {2}, doc_loss: {3}".format(step, loss, labeled_sent_loss, loss - 10 * labeled_sent_loss))

trainStep(False)


