import numpy as np
import torch
import torch.nn as nn

import heapq
from collections import defaultdict

import torch.nn.functional as F

def macro_acc(true_label, pre_label):
    label_2_num = defaultdict(int)
    label_pre_true_num = defaultdict(int)
    label_2_acc = defaultdict(float)

    sz = len(true_label)
    for i in range(sz):
        # true_label[i] = true_label[i].item()

        label_2_num[true_label[i]] += 1
        if (pre_label[i] == true_label[i]):
            label_pre_true_num[true_label[i]] += 1
    for label, num in label_2_num.items():
        label_2_acc[label] = float(label_pre_true_num[label] / num)
    # print(len(label_2_num))

    sum = 0.
    for i, j in label_2_acc.items():
        sum += j

    return sum / len(label_2_acc)


def dtest(te, model, id_space, sem_space, loss_fn):
    sem_space = sem_space.transpose()

    with torch.no_grad():
        model.eval()
        real_label_test = []
        pre_label_test_1 = []
        pre_label_test_2 = []  # hit 2
        pre_label_test_5 = []  # hit 5
        loss_total_test = 0
        for (vx, vy) in te:
            val_vec_y, val_vec_y_neg, val_tag_y = vy
            val_vec_y = val_vec_y.cuda()
            val_vec_y_neg = val_vec_y_neg.cuda()
            vx = vx.cuda()

            vy_pred = model(vx)


            if loss_fn == 'mse':
                loss_fn = nn.MSELoss()
                vloss = loss_fn(vy_pred, val_vec_y)

            if loss_fn == 'margin':
                pos_score = F.cosine_similarity(vy_pred, val_vec_y)
                neg_score = F.cosine_similarity(vy_pred, val_vec_y_neg)
                vloss = torch.mean(torch.max(1.0 - pos_score + neg_score,
                                            torch.zeros_like(pos_score).cuda()))


            loss_total_test += vloss.item()
            #
            val_tag_y = [t.item() for t in val_tag_y]
            real_label_test.extend(val_tag_y)
            #
            vy_pred_cpu = vy_pred.cpu().detach().numpy()
            vsz = len(val_tag_y)
            vtt = np.dot(vy_pred_cpu, sem_space)  # judge by dot Multiplication
            for n in range(vsz):
                e = heapq.nlargest(5, range(len(vtt[n])), vtt[n].take)  # top 5 hit
                vi = 0
                while vi < 5:
                    if (id_space[e[vi]] == val_tag_y[n]):  # pre right
                        break
                    vi += 1
                pre_label_test_1.append(id_space[e[0]])
                pre_label_test_2.append(id_space[e[0]])
                pre_label_test_5.append(id_space[e[0]])

                if (vi <= 1):
                    pre_label_test_2[-1] = val_tag_y[n]
                    pre_label_test_5[-1] = val_tag_y[n]
                elif (vi <= 4):
                    pre_label_test_5[-1] = val_tag_y[n]

        acc_test_1 = macro_acc(real_label_test, pre_label_test_1)
        acc_test_2 = macro_acc(real_label_test, pre_label_test_2)
        acc_test_5 = macro_acc(real_label_test, pre_label_test_5)

        return acc_test_1, acc_test_2, acc_test_5, loss_total_test

