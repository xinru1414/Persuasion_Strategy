
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import *

from Config import ModelConfig, ConversationConfig, FiveFold
from DataLoader import Vocab, dataLoaderANN, dataLoaderUnann
from AttnModel import Request
from MessageLoss import MessageLoss
from PRF import prf, PRResults
import pdb
config = ModelConfig()


def evaluation(without_unann=False):
    global request
    request.eval()
    if without_unann:
        sent_results = PRResults.with_num_of_labels(ConversationConfig.sent_label_num)

        for step, (x, y, l, num, length) in enumerate(loaderDev):
            message_input = Variable(x.type(torch.LongTensor)).cuda()
            message_target = Variable(y.type(torch.FloatTensor)).cuda()
            sentence_label = Variable(l.type(torch.LongTensor)).cuda()

            sentence_out, message_out = request(message_input, num, length)

            loss, label_sent_loss \
                = messageLoss(labeled_doc=None, target1=None, labeled_sent=sentence_out,
                              target2=sentence_label, mode='dev')

            sent_results += prf(predictions=sentence_out, targets=sentence_label, num_labels=ConversationConfig.sent_label_num)

        print("...")
        print(f"Dev: total_loss: {loss}, labeled_sent_loss: {label_sent_loss}")
        for name, result in {'sent': sent_results}.items():
            print(f"   : {name} -- count : {result.count}, acc: {result.accuracy}, p: {result.precision}, r: {result.recall}, f1: {result.f1}")
        print("...")
    else:
        sent_results = PRResults.with_num_of_labels(ConversationConfig.sent_label_num)
        doc_results = PRResults.with_num_of_labels(ConversationConfig.conv_label_num)

        for step, (x, y, l, num, length) in enumerate(loaderDev):
            message_input = Variable(x.type(torch.LongTensor)).cuda()
            message_target = Variable(y.type(torch.FloatTensor)).cuda()
            sentence_label = Variable(l.type(torch.LongTensor)).cuda()

            sentence_out, message_out = request(message_input, num, length)

            loss, label_sent_loss \
                = messageLoss(labeled_doc=message_out, target1=message_target, labeled_sent=sentence_out,
                              target2=sentence_label, mode='dev')
            sent_results += prf(predictions=sentence_out, targets=sentence_label,
                                num_labels=ConversationConfig.sent_label_num)
            doc_results += prf(predictions=message_out, targets=message_target, num_labels=ConversationConfig.conv_label_num)

        print("...")
        print(f"Dev: total_loss: {loss}, labeled_sent_loss: {label_sent_loss}")
        for name, result in {'sent': sent_results, 'conv': doc_results}.items():
            print(f"   : {name} -- count : {result.count}, acc: {result.accuracy}, p: {result.precision}, r: {result.recall}, f1: {result.f1}")
        print("...")

    return sent_results.accuracy, loss


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


def trainStep(path, without_unann=False):
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

            acc, loss = evaluation(without_unann)

            if acc >= max_acc or loss <= min_loss:
                if acc >= max_acc: 
                    print('update!!!')
                    
                    flag = 1
                    torch.save(request, path)
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
            

        request.train()
        
        # if unann_num%1==0 or without_unann:
        ann_message_input = Variable(ann_data[ann_num][0].type(torch.LongTensor)).cuda()
        ann_message_target = Variable(ann_data[ann_num][1].type(torch.FloatTensor)).cuda()
        ann_sentence_label = Variable(ann_data[ann_num][2].type(torch.LongTensor)).cuda()

        ann_sentence_out, ann_message_out = request(ann_message_input, ann_data[ann_num][3], ann_data[ann_num][4])

        ann_num = ann_num - 1
        # else:
        #     ann_message_input = Variable(ann_data[ann_num][0].type(torch.LongTensor)).cuda()
        #     ann_message_target = Variable(ann_data[ann_num][1].type(torch.FloatTensor)).cuda()
        #     ann_sentence_label = Variable(ann_data[ann_num][2].type(torch.LongTensor)).cuda()
        #
        #     ann_sentence_out, ann_message_out = request(ann_message_input, ann_data[ann_num][3], ann_data[ann_num][4])
        #     ann_sentence_label = None
            
            
        # train with unlabled data
        if not without_unann:
            unann_message_input = Variable(unann_data[unann_num][0].type(torch.LongTensor)).cuda()
            unann_message_target = Variable(unann_data[unann_num][1].type(torch.FloatTensor)).cuda()
            #unann_sentence_label = Variable(unann_data[unann_num][2].type(torch.LongTensor)).cuda()
            unann_sentence_out, unann_message_out = request(unann_message_input, unann_data[unann_num][3], unann_data[unann_num][4])
            
            if unann_num%1==0:
                w1 = ann_message_input.shape[0]/(ann_message_input.shape[0] + unann_message_input.shape[0])
            else:
                w1 = 10
            

        unann_num = unann_num - 1

        if without_unann:
            print('no unlabeled sents')
            loss, labeled_sent_loss \
                = messageLoss(labeled_doc=None, target1=None, labeled_sent=ann_sentence_out, target2=ann_sentence_label, w1=0, unlabeled_doc=None, target3=None, mode='train')
        else:
            loss, labeled_sent_loss \
                = messageLoss(labeled_doc=ann_message_out, target1=ann_message_target, labeled_sent=ann_sentence_out, target2=ann_sentence_label, w1=10, unlabeled_doc=unann_message_out, target3=unann_message_target, mode='train')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Train Step {step}: total_loss: {loss}, labeled_sent_loss: {labeled_sent_loss}")


if __name__ == '__main__':
    for i in range(0, 5):
        print(f'training {i} fold')
        dataset_text = FiveFold(i).dataset_text
        dataset_with_annotation = FiveFold(i).dataset_with_annotation

        vocab = Vocab(dataset_with_annotation=dataset_with_annotation, dataset_text=dataset_text)

        print(vocab.vocab_size)

        without_unann = True
        dataSet = dataLoaderANN(vocab, dataset_with_annotation, mode='train')
        dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode='test')
        dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode='dev')

        loaderTrainAnn = Data.DataLoader(dataset=dataSet, batch_size=8, shuffle=True, num_workers=0)
        loaderDev = Data.DataLoader(dataset=dataSet3, batch_size=2, shuffle=False, num_workers=1)

        if not without_unann:
            print('training with unlabled data')
            dataSet4 = dataLoaderUnann(vocab, dataset_text)
            loaderTrainUnann = Data.DataLoader(dataset=dataSet4, batch_size=8, shuffle=True, num_workers=0)
        else:
            print('training with labeled data only')
            dataSet4 = dataLoaderUnann(vocab, dataset_text)
            loaderTrainUnann = Data.DataLoader(dataset=dataSet4, batch_size=8, shuffle=True, num_workers=0)

        request = Request(config, vocab_size=vocab.vocab_size)

        request.cuda()

        messageLoss = MessageLoss(w2=10)
        messageLoss.cuda()

        learning_rate = ModelConfig().learning_rate
        optimizer = torch.optim.Adam(params=request.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        for name, param in request.named_parameters():
            print(name, param.size(), param.requires_grad)

        trainStep(FiveFold(i).save_path, without_unann)


