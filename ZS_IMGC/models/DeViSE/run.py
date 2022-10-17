import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import heapq


from data_reader import DATAReader
from evaluate import dtest, macro_acc

from model import *




def train():




    tr_img = DATAReader(args, 'train_seen')
    tr = DataLoader(tr_img, batch_size=args.batch_size, shuffle=True, num_workers=8)

    model = devise(tr_img.x.shape[1], tr_img.y_vec.shape[1], args.p).cuda()


    optimizer_tag = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wds)
    print('using {} as criterion'.format(args.loss_fn))

    # load unseen testing data
    te_img_unseen = DATAReader(args, 'test_unseen')
    te_unseen = DataLoader(te_img_unseen, batch_size=50, num_workers=8)

    # # load seen testing data
    te_img_seen = DATAReader(args, 'test_seen')
    te_seen = DataLoader(te_img_seen, batch_size=50, num_workers=8)


    print('Begin Training ...')

    for epoch in range(args.epoch_num):
        model.train()
        loss_total = 0

        real_label = []
        pre_label_1 = []
        for i, (x, y) in enumerate(tr, 1):
            vec_y, vec_y_neg, tag_y = y  # vec  tag
            x = x.cuda()
            vec_y = vec_y.cuda()
            vec_y_neg = vec_y_neg.cuda()
            model.zero_grad()
            y_pred = model(x)

            if args.loss_fn == 'mse':
                loss_fn = nn.MSELoss()
                loss = loss_fn(y_pred, vec_y)

            if args.loss_fn == 'margin':
                pos_score = F.cosine_similarity(y_pred, vec_y)
                neg_score = F.cosine_similarity(y_pred, vec_y_neg)

                loss = torch.mean(torch.max(1.0 - pos_score + neg_score,
                                                torch.zeros_like(pos_score).cuda()))


            loss.backward()
            optimizer_tag.step()
            tag_y = [t.item() for t in tag_y]
            real_label.extend(tag_y)  # batch_size
            sz = len(tag_y)
            y_pred_cpu = y_pred.cpu().detach().numpy()
            tt = np.dot(y_pred_cpu, tr_img.all_sem_vec.transpose())  # judge by dot Multiplication
            for n in range(sz):
                e = heapq.nlargest(5, range(len(tt[n])), tt[n].take)
                ii = 0
                while ii < 5:
                    if(tr_img.ids[e[ii]] == tag_y[n]):
                        break
                    ii += 1
                pre_label_1.append(tr_img.ids[e[0]])



            loss_total += loss.item()

        acc_1 = macro_acc(real_label, pre_label_1)  # hit 1
        print('Epoch {:2d}/{:2d}; ----- total_loss:{:06.5f}; macro_acc_1: {:04.2f} -----'.format(epoch, args.epoch_num, loss_total,acc_1*100))

        # gzsl: testing unseen
        acc_test_1, acc_test_2, acc_test_5, loss_total_test = \
            dtest(te_unseen, model, te_img_unseen.ids, te_img_unseen.all_sem_vec, args.loss_fn)

        print('Test ZSL  | Hit 1: {:04.2f}%; Hit 2: {:04.2f}%; Hit 5: {:04.2f}%'.format(acc_test_1 * 100, acc_test_2 * 100, acc_test_5 * 100))


        # gzsl: testing unseen + seen
        acc_unseen, _, _, loss_total_test = \
            dtest(te_unseen, model, te_img_unseen.ids + te_img_seen.ids,
                  np.vstack((te_img_unseen.all_sem_vec, te_img_seen.all_sem_vec)), args.loss_fn)

        acc_seen, _, _, loss_total_test = \
            dtest(te_seen, model, te_img_unseen.ids + te_img_seen.ids,
                  np.vstack((te_img_unseen.all_sem_vec, te_img_seen.all_sem_vec)), args.loss_fn)

        mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print('Test GZSL | Acc Seen: {:04.2f}%; Acc Unseen: {:04.2f}%; Mean: {:04.2f}%'.format(acc_seen * 100,
                                                                                                 acc_unseen * 100,
                                                                                              mean * 100))
        print()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../ZSL_data')
    parser.add_argument('--dataset', default='ImNet_A')

    parser.add_argument('--semantic_type', default='kge', type=str, help='{options: att, w2v, w2v-glove, hie, kge, kge_text, kge_facts, kge_logics}')
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    '''
    Training Parameter
    '''
    parser.add_argument('--loss_fn', default='mse', help='mse, margin')
    parser.add_argument('--p', default=0.5, help='dropout', type=float)
    parser.add_argument('--batch_size', default=64, help='', type=int)
    parser.add_argument('--lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--wds', default=1e-5, help='', type=float)
    parser.add_argument('--epoch_num', default=60, help='', type=int)
    parser.add_argument('--manual_seed', default=12345, help='', type=int)

    args = parser.parse_args()





    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", args.manual_seed)

    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print('using gpu {}'.format(args.gpu))
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.backends.cudnn.deterministic = True


    train()