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
    return count, correct, predict_dict, correct_dict, total_dict


def dicks_to_p_r(correct_dict, predict_dict, correct_total):
    num_of_labels = len(correct_total)
    p, r = 0, 0
    for (u, v) in correct_dict.items():
        if predict_dict[u] != 0:
            p += v / predict_dict[u]
        if correct_total[u] != 0:
            r += v / correct_total[u]
    return p / num_of_labels, r / num_of_labels


def prf_doc(labeled_doc=None, target1=None, labeled_sent=None, target2=None, unlabeled_doc=None, target3=None):
    d_count, d_correct, dpredict_dic, dcorrect_dic, dcorrect_total = p_r_f1(CONV_LABEL_NUM)

    if target1 is not None:
        labeled_doc2 = torch.argmax(F.softmax(labeled_doc, dim=1), dim=1)

        for i in range(0, target1.shape[0]):
            d_count += 1
            dcorrect_total[target1[i].item()] += 1
            dpredict_dic[labeled_doc2[i].item()] += 1
            if labeled_doc2[i] == target1[i].long():
                d_correct += 1
                dcorrect_dic[target1[i].item()] += 1

    ud_count = 0
    if target3 is not None:
        unlabeled_doc2 = torch.argmax(F.softmax(unlabeled_doc, dim=1), dim=1)

        for i in range(0, target3.shape[0]):
            d_count += 1
            ud_count += 1
            dcorrect_total[target3[i].item()] += 1
            dpredict_dic[unlabeled_doc2[i].item()] += 1
            if unlabeled_doc2[i] == target3[i].long():
                d_correct += 1
                dcorrect_dic[target1[i].item()] += 1

    predicted = []
    ground_truth = []
    count, correct, predict_dict, correct_dict, correct_total = p_r_f1(SENT_LABEL_NUM)

    if target2 is not None:
        target2 = target2.view(target2.shape[0] * target2.shape[1])
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

    assert(len(predicted) == len(ground_truth)), 'predicted and ground truth should be the same length'

    dp, dr = dicks_to_p_r(dcorrect_dic, dpredict_dic, dcorrect_total)

    return d_correct, d_count, dp, dr


def prf_sent(labeled_doc=None, target1=None, labeled_sent=None, target2=None, unlabeled_doc=None, target3=None):
    d_count, d_correct, dpredict_dic, dcorrect_dic, dcorrect_total = p_r_f1(CONV_LABEL_NUM)

    if target1 is not None:
        labeled_doc2 = torch.argmax(F.softmax(labeled_doc, dim=1), dim=1)

        for i in range(0, target1.shape[0]):
            d_count += 1
            dcorrect_total[target1[i].item()] += 1
            dpredict_dic[labeled_doc2[i].item()] += 1
            if labeled_doc2[i] == target1[i].long():
                d_correct += 1
                dcorrect_dic[target1[i].item()] += 1

    ud_count = 0
    if target3 is not None:
        unlabeled_doc2 = torch.argmax(F.softmax(unlabeled_doc, dim=1), dim=1)

        for i in range(0, target3.shape[0]):
            d_count += 1
            ud_count += 1
            dcorrect_total[target3[i].item()] += 1
            dpredict_dic[unlabeled_doc2[i].item()] += 1
            if unlabeled_doc2[i] == target3[i].long():
                d_correct += 1
                dcorrect_dic[target1[i].item()] += 1

    predicted = []
    ground_truth = []
    count, correct, predict_dict, correct_dict, correct_total = p_r_f1(SENT_LABEL_NUM)

    if target2 is not None:
        target2 = target2.view(target2.shape[0] * target2.shape[1])
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

    assert(len(predicted) == len(ground_truth)), 'predicted and ground truth should be the same length'

    p, r = dicks_to_p_r(correct_dict, predict_dict, correct_total)

    return correct, count, p, r
