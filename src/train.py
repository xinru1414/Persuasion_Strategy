
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
from sklearn.metrics import cohen_kappa_score

from Config import Config
from DataLoader import Vocab, dataLoaderANN, dataLoaderUnann
from AttnModel import Request
from MessageLoss import MessageLoss

config = Config()

dataset_text = "../data/preprocessed/persuasion_dataset_text.csv"
dataset_with_annotation = "../data/preprocessed/annotation_dataset.csv"


vocab = Vocab(dataset_with_annotation = dataset_with_annotation, dataset_text = dataset_text)

print(vocab.vocab_size)


dataSet = dataLoaderANN(vocab, dataset_with_annotation, mode='train')
dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode='test')
dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode='dev')
dataSet4 = dataLoaderUnann(vocab, dataset_text)

loaderTrainAnn = Data.DataLoader(dataset=dataSet, batch_size=4, shuffle=True, num_workers=0)
loaderTrainUnann = Data.DataLoader(dataset=dataSet4, batch_size=4, shuffle=True, num_workers=0)
loaderDev = Data.DataLoader(dataset=dataSet3, batch_size=2, shuffle=False, num_workers=1)

request = Request(config, vocab_size = vocab.vocab_size)

request.cuda()

messageLoss = MessageLoss(w2=10)
messageLoss.cuda()

learning_rate = 5e-5
optimizer = torch.optim.Adam(params=request.parameters(), lr=learning_rate)
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

        loss, rmse, sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r, dcorrect_dic, dpredict_dic, dcorrect_total, d_correct, d_count, dp, dr \
            = messageLoss(labeled_doc=message_out, target1=message_target, labeled_sent=sentence_out, target2=sentence_label, mode='test')

    if p + r != 0:
        f1 = (2*p*r)/(p+r)
    else:
        f1 = 0

    if dp + dr != 0:
        df1 = (2*dp*dr)/(dp+dr)
    else:
        df1 = 0

    print("...")
    print("Test: total_loss: {0}, score_rmse_loss: {1}, cross_loss: {2}".format(loss, rmse, sent_loss))
    print("   : correct: {0}, count : {1}, acc: {2}".format(correct, count, correct/count))
    print("   : acc: {0}, P: {1}, R: {2}, F1: {3}".format(correct/count, p, r, f1))
    print("   : dcorrect: {0}, dcount : {1}, dacc: {2}".format(d_correct, d_count, d_correct/d_count))
    print("   : dacc: {0}, dP: {1}, dR: {2}, dF1: {3}".format(d_correct/d_count, dp, dr, df1))
    print("...")


def evaluation():
    global request
    request.eval()
    
    correct_ = 0
    count_ = 0
    d_correct_ = 0
    d_count_ = 0
    
    for step, (x, y, l, num, length) in enumerate(loaderDev):
        message_input = Variable(x.type(torch.LongTensor)).cuda()
        message_target = Variable(y.type(torch.FloatTensor)).cuda()
        sentence_label = Variable(l.type(torch.LongTensor)).cuda()

        sentence_out, message_out = request(message_input, num, length)

        loss, rmse, sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r, dcorrect_dic, dpredict_dic, dcorrect_total, d_correct, d_count, dp, dr, x, y \
            = messageLoss(labeled_doc=message_out, target1=message_target, labeled_sent=sentence_out, target2=sentence_label, mode='dev')
        correct_ += correct
        count_ += count

        d_correct_ += d_correct
        d_count_ += d_count

    if p + r != 0:
        f1 = (2*p*r)/(p+r)
    else:
        f1 = 0

    if dp + dr != 0:
        df1 = (2*dp*dr)/(dp+dr)
    else:
        df1 = 0

    print("...")
    print("Dev: total_loss: {0}, score_rmse_loss: {1}, cross_loss: {2}".format(loss, rmse, sent_loss))
    print("   : corrext: {0}, count : {1}, acc: {2}".format(correct_, count_, correct_/count_))
    print("   : dcorrect: {0}, dcount : {1}, dacc: {2}".format(d_correct_, d_count_, d_correct_/d_count_))
    print("   : dacc: {0}, dP: {1}, dR: {2}, dF1: {3}".format(d_correct_/d_count_, dp, dr, df1))
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

            if acc >= max_acc or loss <= min_loss:
                if acc >= max_acc: 
                    print('update!!!')
                    
                    flag = 1
                    torch.save(request, '../model/model_attn_1.pkl')
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
        
        if unann_num%1==0 or without_unann:
            ann_message_input = Variable(ann_data[ann_num][0].type(torch.LongTensor)).cuda()
            ann_message_target = Variable(ann_data[ann_num][1].type(torch.FloatTensor)).cuda()
            ann_sentence_label = Variable(ann_data[ann_num][2].type(torch.LongTensor)).cuda()

            ann_sentence_out, ann_message_out = request(ann_message_input, ann_data[ann_num][3], ann_data[ann_num][4])

            ann_num = ann_num - 1
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
            loss, labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r, dcorrect_dic, dpredict_dic, dcorrect_total, d_correct, d_count, dp, dr, x, y, \
                = messageLoss(labeled_doc=ann_message_out, target1=ann_message_target, labeled_sent=ann_sentence_out, target2=ann_sentence_label, w1=0, unlabeled_doc=None, target3=None, mode='train')
            
        else:
            loss, labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r, dcorrect_dic, dpredict_dic, dcorrect_total, d_correct, d_count, dp, dr, x, y, \
                = messageLoss(labeled_doc=ann_message_out, target1=ann_message_target, labeled_sent=ann_sentence_out, target2=ann_sentence_label, w1=w1, unlabeled_doc=unann_message_out, target3=unann_message_target, mode='train')
            

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Train Step {0}: loss: {1}, labeled_sent_loss: {2}, doc_loss: {3}".format(step, loss, labeled_sent_loss, loss - 10 * labeled_sent_loss))


if __name__ == '__main__':
    trainStep(False)


