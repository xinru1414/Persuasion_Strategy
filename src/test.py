import torch
import torch.utils.data as Data
from torch.autograd import Variable
from Config import ModelConfig, dataset_text, dataset_with_annotation, ConversationConfig
from DataLoader import Vocab, dataLoaderANN, dataLoaderUnann
from MessageLoss import MessageLoss
from nltk import agreement
import numpy as np

from PRF import prf, PRResults

config = ModelConfig()

dataset_text = dataset_text
dataset_with_annotation = dataset_with_annotation

vocab = Vocab(dataset_with_annotation=dataset_with_annotation, dataset_text=dataset_text)

print(vocab.vocab_size)

dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode='test')
dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode='dev')

loaderTest = Data.DataLoader(dataset=dataSet2, batch_size=8, shuffle=False, num_workers=1)
loaderDev = Data.DataLoader(dataset=dataSet3, batch_size=8, shuffle=False, num_workers=1)

request = torch.load('../model/model_attn_1.pkl')
request.cuda()

messageLoss = MessageLoss(w2=10)
messageLoss.cuda()


def testModel():
    global request
    request.eval()

    sent_results = PRResults.with_num_of_labels(ConversationConfig.sent_label_num)
    doc_results = PRResults.with_num_of_labels(ConversationConfig.conv_label_num)

    for step, (x, y, l, num, length) in enumerate(loaderTest):
        message_input = Variable(x.type(torch.LongTensor)).cuda()
        message_target = Variable(y.type(torch.FloatTensor)).cuda()
        sentence_label = Variable(l.type(torch.LongTensor)).cuda()
        sentence_out, message_out = request(message_input, num, length)

        loss, labeled_sent_loss \
            = messageLoss(labeled_doc=message_out, target1=message_target, labeled_sent=sentence_out,
                          target2=sentence_label, mode='test')
        sent_results += prf(predictions=sentence_out, targets=sentence_label, num_labels=ConversationConfig.sent_label_num)
        doc_results += prf(predictions=message_out, targets=message_target, num_labels=ConversationConfig.conv_label_num)

        print(f'x, y {x, y}')
        x = np.stack(x, axis=0).tolist()
        y = np.stack(y, axis=0).tolist()
        print(f'new x y {x, y}')
        taskdata = [[0, str(i), str(x[i])] for i in range(0, len(x))] + [[1, str(i), str(y[i])] for i in range(0, len(y))]
        ratingtask = agreement.AnnotationTask(data=taskdata)
        alpha = ratingtask.kappa()

    print("...")
    print(f"Test: total loss: {loss}, labeled sent loss: {labeled_sent_loss}")
    print(f"   : sent -- count : {sent_results.count}, acc: {sent_results.accuracy}") #, p: {p}, r: {r}, f1: {f1}")
    print(f"   : conv -- count : {doc_results.count}, acc: {doc_results.accuracy}") #, p: {dp}, r: {dr}, f1: {df1}")
    print("...")


if __name__ == '__main__':
    testModel()
