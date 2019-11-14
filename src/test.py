
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from Config import ModelConfig, dataset_text, dataset_with_annotation
from DataLoader import Vocab, dataLoaderANN, dataLoaderUnann
from MessageLoss import MessageLoss
from nltk import agreement
import numpy as np

from PRF import PRF

config = ModelConfig()

dataset_text = dataset_text
dataset_with_annotation = dataset_with_annotation


vocab = Vocab(dataset_with_annotation = dataset_with_annotation, dataset_text = dataset_text)

print(vocab.vocab_size)


dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode='test')
dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode='dev')

loaderTest = Data.DataLoader(dataset=dataSet2, batch_size=8, shuffle=False, num_workers=1)
loaderDev = Data.DataLoader(dataset=dataSet3, batch_size=8, shuffle=False, num_workers=1)

request = torch.load('../model/model_attn_1.pkl')
request.cuda()

messageLoss = MessageLoss(w2 = 10)
messageLoss.cuda()

prf = PRF(w2=10)
prf.cuda()


def testModel():
    global request
    request.eval()
    for step, (x, y, l, num, length) in enumerate(loaderTest):
        message_input = Variable(x.type(torch.LongTensor)).cuda()
        message_target = Variable(y.type(torch.FloatTensor)).cuda()
        sentence_label = Variable(l.type(torch.LongTensor)).cuda()
        sentence_out, message_out = request(message_input, num, length)

        loss, labeled_sent_loss \
            = messageLoss(labeled_doc=message_out, target1=message_target, labeled_sent=sentence_out, target2=sentence_label, mode='test')
        _, _, correct_dict, predict_dict, correct_total, correct, count, p, r, dcorrect_dic, dpredict_dic, dcorrect_total, d_correct, d_count, dp, dr, x, y, \
            = prf(labeled_doc=message_out, target1=message_target, labeled_sent=sentence_out, target2=sentence_label, mode='test')

    if p + r != 0:
        f1 = (2*p*r)/(p+r)
    else:
        f1 = 0

    if dp + dr != 0:
        df1 = (2 * dp * dr) / (dp + dr)
    else:
        df1 = 0

    print(f'x, y {x, y}')
    x = np.stack(x, axis=0).tolist()
    y = np.stack(y, axis=0).tolist()
    print(f'new x y {x, y}')
    taskdata = [[0, str(i), str(x[i])] for i in range(0, len(x))] + [[1, str(i), str(y[i])] for i in range(0, len(y))]
    ratingtask = agreement.AnnotationTask(data=taskdata)
    alpha = ratingtask.kappa()


    print("...")
    print(f"Test: total loss: {loss}, labeled sent loss: {labeled_sent_loss}")
    print(f"    : sent -- correct: {correct}, count : {count}, acc: {correct/count}, p: {p}, r: {r}, f1: {f1}, kappa: {alpha}")
    print(f"    : conv -- correct: {d_correct}, count : {d_count}, dacc: {d_correct/d_count}, p: {dp}, r: {dr}, f1: {df1}")
    print("...")

if __name__ == '__main__':
    testModel()