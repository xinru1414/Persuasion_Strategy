import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import ConversationConfig, zeroed_class_dict
CONV_LABEL_NUM = ConversationConfig.conv_label_num
CONV_PAD_LABEL = ConversationConfig.conv_pad_label
SENT_LABEL_NUM = ConversationConfig.sent_label_num


def p_r_f1(dict_num):
    count = 0
    correct = 0
    predict_dict = zeroed_class_dict(dict_num)
    correct_dict = zeroed_class_dict(dict_num)
    total_dict = zeroed_class_dict(dict_num)
    p = 0
    r = 0
    return count, correct, predict_dict, correct_dict, total_dict, p, r


class PRF(nn.Module):
    def __init__(self, w1=5, w2=10):
        super(PRF, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, labeled_doc=None, target1=None, labeled_sent=None, target2=None, w1=None, unlabeled_doc=None, target3=None, mode='None'):
        if w1 is not None:
            self.w1 = w1

        d_count, d_correct, dpredict_dic, dcorrect_dic, dcorrect_total, dp, dr = p_r_f1(CONV_LABEL_NUM)

        labeled_doc_loss = 0
        if target1 is not None:
            labeled_doc1 = F.log_softmax(labeled_doc, dim=1)
            labeled_doc2 = torch.argmax(F.softmax(labeled_doc, dim=1), dim=1)

            for i in range(0, target1.shape[0]):
                d_count += 1
                dcorrect_total[target1[i].item()] += 1
                dpredict_dic[labeled_doc2[i].item()] += 1
                if labeled_doc2[i] == target1[i].long():
                    d_correct += 1
                    dcorrect_dic[target1[i].item()] += 1
                    labeled_doc_loss += (-1 * labeled_doc1[i][int(target1[i].long().item())])

            if d_count != 0:
                labeled_doc_loss = labeled_doc_loss/d_count

            if type(labeled_doc_loss) != float:
                labeled_doc_loss = labeled_doc_loss.cpu().detach().numpy()

        unlabeled_doc_loss = 0
        ud_count = 0
        if target3 is not None:
            unlabeled_doc1 = F.log_softmax(unlabeled_doc, dim=1)
            unlabeled_doc2 = torch.argmax(F.softmax(unlabeled_doc, dim=1), dim=1)

            for i in range(0, target3.shape[0]):
                d_count += 1
                ud_count += 1
                dcorrect_total[target3[i].item()] += 1
                dpredict_dic[unlabeled_doc2[i].item()] += 1
                if unlabeled_doc2[i] == target3[i].long():
                    d_correct += 1
                    dcorrect_dic[target1[i].item()] += 1
                    unlabeled_doc_loss += (-1 * unlabeled_doc1[i][int(target3[i].long().item())])

            if ud_count != 0:
                unlabeled_doc_loss = unlabeled_doc_loss / ud_count

            if type(unlabeled_doc_loss) != float:
                unlabeled_doc_loss = unlabeled_doc_loss.cpu().detach().numpy()

        labeled_sent_loss = 0
        predicted = []
        ground_truth = []
        count, correct, predict_dict, correct_dict, correct_total, p, r = p_r_f1(SENT_LABEL_NUM)

        if target2 is not None:
            target2 = target2.view(target2.shape[0] * target2.shape[1])
            labeled_sent1 = F.log_softmax(labeled_sent, dim=1)
            labeled_sent2 = torch.argmax(F.softmax(labeled_sent, dim=1), dim=1)

            for i in range(0, target2.shape[0]):
                if target2[i] == CONV_PAD_LABEL:
                    continue
                else:
                    predicted.append(labeled_sent2[i].cpu().detach().numpy())
                    ground_truth.append(target2[i].cpu().detach().numpy())
                    count += 1
                    correct_total[target2[i].item()] += 1
                    predict_dict[labeled_sent2[i].item()] += 1
                    if labeled_sent2[i] == target2[i]:
                        correct += 1
                        correct_dict[target2[i].item()] += 1
                    labeled_sent_loss += (-1 * labeled_sent1[i][target2[i].item()])

            if count != 0:
                labeled_sent_loss = labeled_sent_loss/count
        assert(len(predicted) == len(ground_truth)), 'predicted and ground truth should be the same length'

        if mode != 'train':
            for (u, v) in correct_dict.items():
                if predict_dict[u] == 0:
                    temp = 0
                else:
                    temp = v/predict_dict[u]
                if correct_total[u] != 0:
                    temp2 = v/correct_total[u]
                else:
                    temp2 = 0
                p += temp
                r += temp2
            for (u,v) in dcorrect_dic.items():
                if dpredict_dic[u] == 0:
                    temp = 0
                else:
                    temp = v/dpredict_dic[u]
                if dcorrect_total[u] != 0:
                    temp2 = v/dcorrect_total[u]
                else:
                    temp2 = 0
                dp += temp
                dr += temp2

        if mode == 'train':
            loss = self.w1 * (labeled_doc_loss + unlabeled_doc_loss) + self.w2 * labeled_sent_loss
            return loss, labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p/SENT_LABEL_NUM, r/SENT_LABEL_NUM, \
                   dcorrect_dic, dpredict_dic, dcorrect_total, d_correct, d_count, dp/CONV_LABEL_NUM, dr/CONV_LABEL_NUM, predicted, ground_truth
        else:
            loss = labeled_doc_loss + self.w2 * labeled_sent_loss.cpu().detach().numpy()
            return loss, labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p/SENT_LABEL_NUM, r/SENT_LABEL_NUM,\
                   dcorrect_dic, dpredict_dic, dcorrect_total, d_correct, d_count, dp/CONV_LABEL_NUM, dr/CONV_LABEL_NUM, predicted, ground_truth
