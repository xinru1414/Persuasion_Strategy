import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from Config import ConversationConfig
CONV_LABEL_NUM = ConversationConfig.conv_label_num
CONV_PAD_LABEL = ConversationConfig.conv_pad_label
SENT_LABEL_NUM = ConversationConfig.sent_label_num


class MessageLoss(nn.Module):
    def __init__(self, w1=5, w2=10):
        super(MessageLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        
    def forward(self, labeled_doc=None, target1=None, labeled_sent=None, target2=None, w1=None, unlabeled_doc=None, target3=None, mode='None'):
        if w1 is not None:
            self.w1 = w1

        d_count = 0
        labeled_doc_loss = 0
        if target1 is not None:
            labeled_doc1 = F.log_softmax(labeled_doc, dim=1)

            for i in range(0, target1.shape[0]):
                d_count += 1
                labeled_doc_loss += (-1 * labeled_doc1[i][int(target1[i].long().item())])

            if d_count != 0:
                labeled_doc_loss = labeled_doc_loss/d_count

            if type(labeled_doc_loss) != float:
                labeled_doc_loss = labeled_doc_loss.cpu().detach().numpy()

        unlabeled_doc_loss = 0
        ud_count = 0
        if target3 is not None:
            unlabeled_doc1 = F.log_softmax(unlabeled_doc, dim=1)

            for i in range(0, target3.shape[0]):
                ud_count += 1
                unlabeled_doc_loss += (-1 * unlabeled_doc1[i][int(target3[i].long().item())])

            if ud_count != 0:
                unlabeled_doc_loss = unlabeled_doc_loss / ud_count

            if type(unlabeled_doc_loss) != float:
                unlabeled_doc_loss = unlabeled_doc_loss.cpu().detach().numpy()
            
        labeled_sent_loss = 0
        sent_count = 0
        #predicted = []
        #ground_truth = []
        #count, correct, predict_dict, correct_dict, correct_total = p_r_f1(SENT_LABEL_NUM)
        
        if target2 is not None:
            print(f'labeled sent {labeled_sent.shape}')
            breakpoint()
            target2 = target2.view(target2.shape[0] * target2.shape[1])
            labeled_sent1 = F.log_softmax(labeled_sent, dim=1)
            labeled_sent = torch.argmax(F.softmax(labeled_sent, dim=1), dim=1)
            # print(f'labeld sent {labeled_sent}')

            for i in range(0, target2.shape[0]):
                if target2[i] == CONV_PAD_LABEL:
                    continue
                else:
                    #predicted.append(labeled_sent2[i].cpu().detach().numpy())
                    #ground_truth.append(target2[i].cpu().detach().numpy())
                    sent_count += 1
                    labeled_sent_loss += (-1 * labeled_sent1[i][target2[i].item()])
            if sent_count != 0:
                labeled_sent_loss = labeled_sent_loss/sent_count



        loss = self.w1 * (labeled_doc_loss + unlabeled_doc_loss) + self.w2 * labeled_sent_loss
        return loss, labeled_sent_loss
