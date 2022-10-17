import numpy as np
import random
import torch
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.init as init
from sklearn.metrics.pairwise import cosine_similarity
import sys
import argparse
import os
import os.path as osp
import json
import shutil

from functions import *
from data_loader import Extractor_generate, centroid_generate, train_generate_decription
from networks import Extractor, Generator, Discriminator


def calc_gradient_penalty(netD, real_data, fake_data, batchsize, centroid_matrix):
    alpha = torch.rand(batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    _, disc_interpolates, _ = netD(interpolates, centroid_matrix)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10 #opt.GP_LAMBDA
    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)

def reset_grad(nets):
    for net in nets:
        net.zero_grad()




class Trainer(object):

    def __init__(self, args):
        super(Trainer, self).__init__()
        for k, v in vars(args).items():setattr(self, k, v)
        self.args = args


        self.train_tasks = json.load(open(os.path.join(args.data_path, 'datasplit', 'train_tasks.json')))
        self.rel2id = json.load(open(os.path.join(args.data_path, 'relation2ids')))

        self.rela_matrix = load_semantic_embed(args.data_path, args.dataset, args.semantic_type)
        self.args.input_dim = self.rela_matrix.shape[1]


        self.ent2id = json.load(open(os.path.join(args.data_path, 'entity2id')))

        print('##LOADING CANDIDATES ENTITIES##')
        self.rel2candidates = json.load(open(os.path.join(args.data_path, 'rel2candidates_all.json')))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(os.path.join(args.data_path, 'e1rel_e2_all.json')))


        noises = Variable(torch.randn(args.test_sample, args.noise_dim)).cuda()
        self.test_noises = 0.1 * noises

        self.label_num = len(self.train_tasks.keys())

        self.rela2label = dict()
        rela_sorted = sorted(list(self.train_tasks.keys()))
        for i, rela in enumerate(rela_sorted):
            self.rela2label[rela] = int(i)

        print('##LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if args.load_trained_embed:
            self.load_embed()
        else:
            self.read_embed()


        self.num_symbols = len(self.symbol2id.keys()) - 1
        print("num symbols:", self.num_symbols)
        self.pad_id = self.num_symbols

        print('##DEFINE FEATURE EXTRACTOR')
        self.Extractor = Extractor(args.embed_dim, self.num_symbols, embed=self.symbol2vec)
        self.Extractor.cuda()
        self.Extractor.apply(weights_init)
        self.E_parameters = filter(lambda p: p.requires_grad, self.Extractor.parameters())
        self.optim_E = optim.Adam(self.E_parameters, lr=args.lr_E)

        print('##DEFINE GENERATOR')
        self.Generator = Generator(self.args)
        self.Generator.cuda()
        self.Generator.apply(weights_init)
        self.G_parameters = filter(lambda p: p.requires_grad, self.Generator.parameters())
        self.optim_G = optim.Adam(self.G_parameters, lr=args.lr_G, betas=(0.5, 0.9))
        self.scheduler_G = optim.lr_scheduler.MultiStepLR(self.optim_G, milestones=[4000], gamma=0.2)

        print('##DEFINE DISCRIMINATOR')
        self.Discriminator = Discriminator(self.args)
        self.Discriminator.cuda()
        self.Discriminator.apply(weights_init)
        self.D_parameters = filter(lambda p: p.requires_grad, self.Discriminator.parameters())
        self.optim_D = optim.Adam(self.D_parameters, lr=args.lr_D, betas=(0.5, 0.9))
        self.scheduler_D = optim.lr_scheduler.MultiStepLR(self.optim_D, milestones=[20000], gamma=0.2)

        self.num_ents = len(self.ent2id.keys())

        print('##BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=args.max_neighbor)



    def ensure_path(self, path):
        print(path)
        if osp.exists(path):
            if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
                shutil.rmtree(path)
                os.mkdir(path)
        else:
            os.mkdir(path)

    def load_embed(self):

        symbol_id = {}

        print('##LOADING PRE-TRAINED EMBEDDING')
        if self.args.embed_model in ['DistMult', 'TransE']:
            embed_all = np.load(os.path.join(self.args.data_path, self.args.embed_model + '_embed.npz'))
            ent_embed = embed_all['eM']
            rel_embed = embed_all['rM']
            print('    ent_embed shape is {}, the number of entity is {}'.format(ent_embed.shape,
                                                                                 len(self.ent2id.keys())))
            print('    rel_embed shape is {}, the number of relation is {}'.format(rel_embed.shape,
                                                                                   len(self.rel2id.keys())))

            i = 0
            embeddings = []
            for key in self.rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[self.rel2id[key], :]))

            for key in self.ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[self.ent2id[key], :]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)



            np.savez(os.path.join(self.args.data_path, 'Embed_used', self.args.embed_model), embeddings)
            json.dump(symbol_id, open(os.path.join(self.args.data_path, 'Embed_used', self.args.embed_model + '2id'), 'w'))

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def read_embed(self):
        symbol_id = json.load(open(
                os.path.join(self.args.data_path, 'Embed_used', self.args.embed_model + '2id')))
        embeddings = np.load(os.path.join(self.args.data_path, 'Embed_used', self.args.embed_model + '.npz'))['arr_0']

        self.symbol2id = symbol_id
        self.symbol2vec = embeddings

    #  build neighbor connection
    def build_connection(self, max_=100):

        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        # rel_list = list()
        with open(os.path.join(self.args.data_path, 'path_graph')) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                self.e1_rele2[e2].append((self.symbol2id[rel], self.symbol2id[e1]))

        # print("path graph relations:", len(set(rel_list)))
        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]
        # print(self.connections[0])
        # json.dump(degrees, open(self.dataset + '/degrees', 'w'))
        # assert 1==2

        return degrees

    def save_pretrain(self):
        torch.save(self.Extractor.state_dict(), os.path.join(self.args.data_path, 'FE_models_trained', self.args.embed_model + '_Extractor'))


    def load_pretrain(self):
        self.Extractor.load_state_dict(torch.load(os.path.join(self.args.data_path, 'FE_models_trained', self.args.embed_model + '_Extractor'), map_location=lambda storage, loc: storage.cuda(self.args.gpu)))
        self.Extractor.eval()




    def save_model(self):

        path = self.args.save_path

        torch.save(self.Generator.state_dict(), os.path.join(path, self.args.embed_model + '_Generator'))
        torch.save(self.Discriminator.state_dict(), os.path.join(path, self.args.embed_model + '_Discriminator'))

    def load_model(self):
        self.Generator.load_state_dict(torch.load(os.path.join(self.args.save_path, self.args.embed_model + '_Generator')))
        self.Discriminator.load_state_dict(torch.load(os.path.join(self.args.save_path, self.args.embed_model + '_Discriminator')))

    def get_meta(self, left, right):
        left_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def pretrain_Extractor(self):
        print('\n##PRETRAINING FEATURE EXTRACTOR ....')
        # self.ensure_path(self.args.save_path)

        pretrain_losses = deque([], 100)

        i = 0
        for data in Extractor_generate(self.args.manual_seed, self.args.data_path, self.train_tasks, self.pretrain_batch_size, self.symbol2id, self.ent2id,
                                       self.e1rel_e2, self.pretrain_few, self.pretrain_subepoch):
            i += 1

            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).cuda()
            query = Variable(torch.LongTensor(query)).cuda()
            false = Variable(torch.LongTensor(false)).cuda()

            query_ep, query_scores = self.Extractor(query, support, query_meta, support_meta)
            false_ep, false_scores = self.Extractor(false, support, false_meta, support_meta)

            margin_ = query_scores - false_scores
            pretrain_loss = F.relu(self.args.pretrain_margin - margin_).mean()

            self.optim_E.zero_grad()
            pretrain_loss.backward()
            # self.scheduler.step()
            pretrain_losses.append(pretrain_loss.item())

            if i % self.args.pretrain_loss_every == 0:
                print("Step: %d, Feature Extractor Pretraining loss: %.10f" % (i, np.mean(pretrain_losses)))

            self.optim_E.step()

            if i > self.args.pretrain_times:
                break



        self.save_pretrain()
        print('SAVE FEATURE EXTRACTOR PRETRAINING MODEL!!!')

    def train(self):
        print('\n##START ADVERSARIAL TRAINING...')

        # Pretraining step to obtain reasonable real data embeddings
        if self.args.pretrain_feature_extractor:
            self.pretrain_Extractor()
            print('Finish Pretraining!\n')

        self.load_pretrain()


        self.centroid_matrix = torch.zeros((len(self.train_tasks), self.args.ep_dim))
        self.centroid_matrix = self.centroid_matrix.cuda()

        for relname in self.train_tasks.keys():
            query, query_left, query_right, label_id = centroid_generate(self.args.data_path, relname, self.symbol2id,
                                                                         self.ent2id, self.train_tasks, self.rela2label)
            query_meta = self.get_meta(query_left, query_right)
            query = Variable(torch.LongTensor(query)).cuda()
            query_ep, _ = self.Extractor(query, query, query_meta, query_meta)
            self.centroid_matrix[label_id] = query_ep.data.mean(dim=0)
        self.centroid_matrix = Variable(self.centroid_matrix)

        best_hits10 = 0.0

        D_every = self.args.D_epoch * self.args.loss_every
        D_losses = deque([], D_every)
        D_real_losses, D_real_class_losses, D_fake_losses, D_fake_class_losses \
            = deque([], D_every), deque([], D_every), deque([], D_every), deque([], D_every)

        # loss_G_fake + loss_G_class + loss_VP
        G_every = self.args.G_epoch * self.args.loss_every
        G_losses = deque([], G_every)
        G_fake_losses, G_class_losses, G_VP_losses, G_real_class_losses \
            = deque([], G_every), deque([], G_every), deque([], G_every), deque([], G_every)

        G_data = train_generate_decription(self.args.data_path, self.train_tasks, self.args.G_batch_size, self.symbol2id, self.ent2id,
                                           self.e1rel_e2, self.rel2id, self.args, self.rela2label, self.rela_matrix)

        nets = [self.Generator, self.Discriminator]
        reset_grad(nets)

        for epoch in range(1, (self.args.train_times+1)):

            # train Discriminator
            self.Discriminator.train()
            self.Generator.eval()
            for _ in range(self.args.D_epoch):  # D_epoch = 5
                ### Discriminator real part
                D_descriptions, query, query_left, query_right, D_false, D_false_left, D_false_right, D_labels = G_data.__next__()

                # real part
                query_meta = self.get_meta(query_left, query_right)
                query = Variable(torch.LongTensor(query)).cuda()
                D_real, _ = self.Extractor(query, query, query_meta, query_meta)

                # fake part
                noises = Variable(torch.randn(len(query), self.noise_dim)).cuda()
                D_descriptions = Variable(torch.FloatTensor(D_descriptions)).cuda()
                D_fake = self.Generator(D_descriptions, noises)

                # neg part
                D_false_meta = self.get_meta(D_false_left, D_false_right)
                D_false = Variable(torch.LongTensor(D_false)).cuda()
                D_neg, _ = self.Extractor(D_false, D_false, D_false_meta, D_false_meta)

                # generate Discriminator part vector
                centroid_matrix_ = self.centroid_matrix  # gaussian_noise(self.centroid_matrix)
                _, D_real_decision, D_real_class = self.Discriminator(D_real.detach(), centroid_matrix_)
                _, D_fake_decision, D_fake_class = self.Discriminator(D_fake.detach(), centroid_matrix_)
                _, _, D_neg_class = self.Discriminator(D_neg.detach(), self.centroid_matrix)

                # real adversarial training loss
                loss_D_real = -torch.mean(D_real_decision)

                # adversarial training loss
                loss_D_fake = torch.mean(D_fake_decision)

                # real classification loss
                D_real_scores = D_real_class[range(len(query)), D_labels]
                D_neg_scores = D_neg_class[range(len(query)), D_labels]
                D_margin_real = D_real_scores - D_neg_scores
                loss_rela_class = F.relu(self.args.pretrain_margin - D_margin_real).mean()

                # fake classification loss
                D_fake_scores = D_fake_class[range(len(query)), D_labels]
                D_margin_fake = D_fake_scores - D_neg_scores
                loss_fake_class = F.relu(self.args.pretrain_margin - D_margin_fake).mean()

                grad_penalty = calc_gradient_penalty(self.Discriminator, D_real.data, D_fake.data, len(query),
                                                     self.centroid_matrix)

                loss_D = loss_D_real + 0.5 * loss_rela_class + loss_D_fake + grad_penalty + 0.5 * loss_fake_class

                # D_real_losses, D_real_class_losses, D_fake_losses, D_fake_class_losses
                D_losses.append(loss_D.item())
                D_real_losses.append(loss_D_real.item())
                D_real_class_losses.append(loss_rela_class.item())
                D_fake_losses.append(loss_D_fake.item())
                D_fake_class_losses.append(loss_fake_class.item())

                loss_D.backward()
                self.scheduler_D.step()
                self.optim_D.step()
                reset_grad(nets)

            # train Generator
            self.Discriminator.eval()
            self.Generator.train()
            for _ in range(self.args.G_epoch):  # G_epoch = 1

                G_descriptions, query, query_left, query_right, G_false, G_false_left, G_false_right, G_labels = G_data.__next__()

                # G sample
                noises = Variable(torch.randn(len(query), self.args.noise_dim)).cuda()
                G_descriptions = Variable(torch.FloatTensor(G_descriptions)).cuda()
                G_sample = self.Generator(G_descriptions, noises)  # to train G

                # real data
                query_meta = self.get_meta(query_left, query_right)
                query = Variable(torch.LongTensor(query)).cuda()
                G_real, _ = self.Extractor(query, query, query_meta, query_meta)

                # This negative for classification loss
                G_false_meta = self.get_meta(G_false_left, G_false_right)
                G_false = Variable(torch.LongTensor(G_false)).cuda()
                G_neg, _ = self.Extractor(G_false, G_false, G_false_meta,
                                          G_false_meta)  # just use Extractor to generate ep vector

                # generate Discriminator part vector
                centroid_matrix_ = self.centroid_matrix
                _, G_decision, G_class = self.Discriminator(G_sample, centroid_matrix_)
                _, _, G_real_class = self.Discriminator(G_real.detach(), centroid_matrix_)
                _, _, G_neg_class = self.Discriminator(G_neg.detach(), centroid_matrix_)

                # adversarial training loss
                loss_G_fake = - torch.mean(G_decision)

                # G sample (fake) classification loss
                G_scores = G_class[range(len(query)), G_labels]
                G_neg_scores = G_neg_class[range(len(query)), G_labels]
                G_margin_ = G_scores - G_neg_scores
                loss_G_class = F.relu(self.args.pretrain_margin - G_margin_).mean()

                # real classification loss
                G_real_scores = G_real_class[range(len(query)), G_labels]
                G_margin_real = G_real_scores - G_neg_scores
                loss_rela_class_ = F.relu(self.args.pretrain_margin - G_margin_real).mean()

                # Visual Pivot Regularization
                count = 0
                loss_VP = Variable(torch.Tensor([0.0])).cuda()
                for i in range(len(self.train_tasks.keys())):
                    sample_idx = (np.array(G_labels) == i).nonzero()[0]
                    count += len(sample_idx)
                    if len(sample_idx) == 0:
                        loss_VP += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        loss_VP += (G_sample_cls.mean(dim=0) - self.centroid_matrix[i]).pow(2).sum().sqrt()
                assert count == len(query)
                loss_VP *= float(1.0 / self.args.gan_batch_rela)

                # ||W||_2 regularization
                reg_loss = Variable(torch.Tensor([0.0])).cuda()
                if self.args.REG_W != 0:
                    for name, p in self.Generator.named_parameters():
                        if 'weight' in name:
                            reg_loss += p.pow(2).sum()
                    reg_loss.mul_(self.args.REG_W)

                # ||W_z||21 regularization, make W_z sparse
                reg_Wz_loss = Variable(torch.Tensor([0.0])).cuda()
                if self.args.REG_Wz != 0:
                    Wz = self.Generator.fc1.weight
                    reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(self.args.REG_Wz)

                # Generator loss function
                loss_G = loss_G_fake + loss_G_class + 3.0 * loss_VP  # + reg_Wz_loss + reg_loss

                # G_fake_losses, G_class_losses, G_VP_losses
                G_losses.append(loss_G.item())
                G_fake_losses.append(loss_G_fake.item())
                G_class_losses.append(loss_G_class.item())
                G_real_class_losses.append(loss_rela_class_.item())
                G_VP_losses.append(loss_VP.item())

                loss_G.backward()
                self.scheduler_G.step()
                self.optim_G.step()
                reset_grad(nets)

            if epoch % self.args.loss_every == 0:
                D_screen = [np.mean(D_real_losses), np.mean(D_real_class_losses), np.mean(D_fake_losses),
                            np.mean(D_fake_class_losses)]
                G_screen = [np.mean(G_fake_losses), np.mean(G_class_losses), np.mean(G_real_class_losses),
                            np.mean(G_VP_losses)]
                print("Epoch: %d, D_loss: %.2f [%.2f, %.2f, %.2f, %.2f], G_loss: %.2f [%.2f, %.2f, %.2f, %.2f]" \
                      % (
                      epoch, np.mean(D_losses), D_screen[0], D_screen[1], D_screen[2], D_screen[3], np.mean(G_losses),
                      G_screen[0], G_screen[1], G_screen[2], G_screen[3]))

            # D_screen = [np.mean(D_real_losses), np.mean(D_real_class_losses), np.mean(D_fake_losses),
            #             np.mean(D_fake_class_losses)]
            # G_screen = [np.mean(G_fake_losses), np.mean(G_class_losses), np.mean(G_real_class_losses),
            #             np.mean(G_VP_losses)]
            # print("Epoch: %d, D_loss: %.2f [%.2f, %.2f, %.2f, %.2f], G_loss: %.2f [%.2f, %.2f, %.2f, %.2f]" \
            #       % (
            #           epoch, np.mean(D_losses), D_screen[0], D_screen[1], D_screen[2], D_screen[3], np.mean(G_losses),
            #           G_screen[0], G_screen[1], G_screen[2], G_screen[3]))

            if epoch >= 1000 and epoch % self.args.eval_every == 0:
                self.eval(mode='test', epoch=epoch)
                # self.save_model()



    def eval(self, mode='dev', epoch=0):
        self.Generator.eval()
        self.Discriminator.eval()
        # self.Extractor.eval()
        symbol2id = self.symbol2id

        print('##EVALUATING ON %s DATA' % mode.upper())
        # test_candidates = json.load(open(self.args.data_path + "/test_candidates_sub_10.json"))
        test_candidates = json.load(open(self.args.data_path + "/test_candidates.json"))

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []


        for query_ in sorted(test_candidates.keys()):


            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            description = self.rela_matrix[self.rel2id[query_]]
            description = np.expand_dims(description, axis=0)
            descriptions = np.tile(description, (self.args.test_sample, 1))
            descriptions = Variable(torch.FloatTensor(descriptions)).cuda()
            relation_vecs = self.Generator(descriptions, self.test_noises)
            relation_vecs = relation_vecs.data.cpu().numpy()

            for e1_rel, tail_candidates in test_candidates[query_].items():
                if self.args.dataset == "NELL":
                    head, rela, _ = e1_rel.split('\t')
                elif self.args.dataset == "Wiki":
                    head, rela = e1_rel.split('\t')

                true = tail_candidates[0]
                query_pairs = []
                if head not in symbol2id or true not in symbol2id:
                    continue
                query_pairs.append([symbol2id[head], symbol2id[true]])


                query_left = []
                query_right = []
                query_left.append(self.ent2id[head])
                query_right.append(self.ent2id[true])

                for tail in tail_candidates[1:]:
                    if tail not in symbol2id:
                        continue
                    query_pairs.append([symbol2id[head], symbol2id[tail]])

                    query_left.append(self.ent2id[head])
                    query_right.append(self.ent2id[tail])

                query = Variable(torch.LongTensor(query_pairs)).cuda()


                query_meta = self.get_meta(query_left, query_right)
                candidate_vecs, _ = self.Extractor(query, query, query_meta, query_meta)

                candidate_vecs.detach()
                candidate_vecs = candidate_vecs.data.cpu().numpy()

                # dot product
                # scores = candidate_vecs.dot(relation_vecs.transpose())

                # cosine similarity
                scores = cosine_similarity(candidate_vecs, relation_vecs)

                scores = scores.mean(axis=1)

                assert scores.shape == (len(query_pairs),)

                sort = list(np.argsort(scores))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)



        print('\n############   ' + mode + ' ' + str(epoch) + '    #############')
        print('HITS10: {:.3f}, HITS5: {:.3f}, HITS1: {:.3f}, MAP: {:.3f}'.format(np.mean(hits10),
                                                                                 np.mean(hits5),
                                                                                 np.mean(hits1),
                                                                                 np.mean(mrr)))
        print('###################################')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../ZSL_data", type=str)

    parser.add_argument("--dataset", default="NELL", type=str)
    parser.add_argument("--embed_model", default='TransE', type=str)

    # embedding dimension
    parser.add_argument("--embed_dim", default=100, type=int, help='dimension of triple embedding')
    parser.add_argument("--ep_dim", default=200, type=int, help='dimension of entity pair embedding')
    parser.add_argument("--fc1_dim", default=250, type=int, help='dimension of hidden units in generator')
    parser.add_argument("--noise_dim", default=15, type=int)

    # feature extractor pretraining related
    parser.add_argument("--pretrain_batch_size", default=64, type=int)
    parser.add_argument("--pretrain_few", default=30, type=int)
    parser.add_argument("--pretrain_subepoch", default=20, type=int)
    parser.add_argument("--pretrain_margin", default=10.0, type=float, help='pretraining margin loss')
    parser.add_argument("--pretrain_times", default=16000, type=int, help='total training steps for pretraining')
    parser.add_argument("--pretrain_loss_every", default=500, type=int)

    # adversarial training related
    # batch size
    parser.add_argument("--D_batch_size", default=256, type=int)
    parser.add_argument("--G_batch_size", default=256, type=int)
    parser.add_argument("--gan_batch_rela", default=2, type=int)
    # learning rate
    parser.add_argument("--lr_G", default=0.0001, type=float)
    parser.add_argument("--lr_D", default=0.0001, type=float)
    parser.add_argument("--lr_E", default=0.0005, type=float)
    # training times
    parser.add_argument("--train_times", default=8000, type=int)
    parser.add_argument("--D_epoch", default=5, type=int)
    parser.add_argument("--G_epoch", default=1, type=int)
    # log
    parser.add_argument("--loss_every", default=50, type=int)
    parser.add_argument("--eval_every", default=200, type=int)
    # hyper-parameter
    parser.add_argument("--test_sample", default=20, type=int, help='number of synthesized samples')
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument('--REG_W', default=0.001, type=float)
    parser.add_argument('--REG_Wz', default=0.0001, type=float)
    parser.add_argument("--max_neighbor", default=50, type=int, help='neighbor number of each entity')
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--semantic_type", default='rdfs', help='{options: text, rdfs, rdfs_hie, rdfs_cons, rdfs_text}')
    # switch
    parser.add_argument("--pretrain_feature_extractor", action='store_true')
    parser.add_argument("--load_trained_embed", action='store_true', help='load well trained kg embeddings, such as TransE')


    parser.add_argument("--manual_seed", type=int, default=6096)
    parser.add_argument('--gpu', type=int, default=2, help='device to use for iterate data, -1 means cpu [default: 0]')

    args = parser.parse_args()

    args.data_path = os.path.join(args.data_dir, args.dataset)

    args.save_path = os.path.join(args.data_path, 'expri_data', 'models_train')

    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)

    print("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v) + ';')
    print("----------------------------")

    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print('using gpu {}'.format(args.gpu))
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.backends.cudnn.deterministic = True
    else:
        print("GPU is not available!")



    trainer = Trainer(args)
    trainer.train()
    # trainer.test_()

