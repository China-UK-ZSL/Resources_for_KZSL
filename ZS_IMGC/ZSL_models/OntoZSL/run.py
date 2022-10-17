from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import model
import classifier_pretrain, classifier_cls
import util



class Runner:
    def __init__(self, args):
        self.args = args

        print('============ Params ============')
        print('\n'.join('%s: %s; ' % (k, str(v)) for k, v
                        in sorted(dict(vars(self.args)).items())))
        print('============================================')


        # load data
        self.data = util.DATA_LOADER(self.args)
        self.feat_dim = self.data.feat_dim
        self.sem_dim = self.data.sem_dim
        self.semantic = self.data.semantic
        print("Training samples: ", self.data.ntrain)  # number of training samples

        args.feat_dim = self.feat_dim
        args.sem_dim = self.sem_dim

        # initialize generator and discriminator
        self.netG = model.MLP_G(args)
        self.netD = model.MLP_CRITIC(args)
        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta, 0.999))
        # classification loss
        self.cls_criterion = nn.NLLLoss()  # cross entropy loss

        self.input_fea = torch.FloatTensor(args.batch_size, self.feat_dim)  # (64, 2048)
        self.input_sem = torch.FloatTensor(args.batch_size, self.data.sem_dim)  # (64, 500)
        self.noise = torch.FloatTensor(args.batch_size, args.noise_size)  # (64, 500)
        self.input_label = torch.LongTensor(args.batch_size)

        if self.args.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.input_fea = self.input_fea.cuda()
            self.noise, self.input_sem = self.noise.cuda(), self.input_sem.cuda()
            self.cls_criterion.cuda()
            self.input_label = self.input_label.cuda()

        # train a classifier on seen classes, obtain \theta of Equation (4)
        self.pretrain_cls = classifier_pretrain.CLASSIFIER(self.data.train_seen_feature,
                                                      util.map_label(self.data.train_seen_label, self.data.seenclasses),
                                                      self.data.seenclasses.size(0), self.feat_dim, args.cuda, 0.001, 0.5,
                                                      100, 2 * args.batch_size)

        # freeze the classifier during the optimization
        for p in self.pretrain_cls.model.parameters():  # set requires_grad to False
            p.requires_grad = False


    def sample(self):
        batch_feature, batch_label, batch_sem = self.data.next_batch(args.batch_size)
        self.input_fea.copy_(batch_feature)
        self.input_sem.copy_(batch_sem)
        self.input_label.copy_(util.map_label(batch_label, self.data.seenclasses))


    def generate_syn_feature(self, num):
        classes = self.data.unseenclasses
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass * num, self.feat_dim)
        syn_label = torch.LongTensor(nclass * num)
        syn_sem = torch.FloatTensor(num, self.sem_dim)
        syn_noise = torch.FloatTensor(num, args.noise_size)
        if self.args.cuda:
            syn_sem = syn_sem.cuda()
            syn_noise = syn_noise.cuda()
        for i in range(nclass):
            iclass = classes[i]
            iclass_sem = self.semantic[iclass]
            syn_sem.copy_(iclass_sem.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output = self.netG(Variable(syn_noise, volatile=True), Variable(syn_sem, volatile=True))
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)
        return syn_feature, syn_label



    # the last item of equation (2)
    def calc_gradient_penalty(self, real_data, fake_data, input_sem):
        # print real_data.size()
        alpha = torch.rand(args.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if args.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if args.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, Variable(input_sem))

        ones = torch.ones(disc_interpolates.size())
        if args.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        # args.GP_Weight = 10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.GP_weight
        return gradient_penalty





    def train(self):
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.args.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for epoch in range(args.epoch):
            for i in range(0, self.data.ntrain, args.batch_size):
                # print("batch...", i)
                # iteratively train the generator and discriminator
                for p in self.netD.parameters():
                    p.requires_grad = True

                # DISCRIMINATOR
                # args.critic_iter = 5, following WGAN-GP
                for iter_d in range(args.critic_iter):
                    self.sample()  # sample by batch
                    self.netD.zero_grad()
                    # torch.gt: compare the 'input_res[1]' and '0' element by element
                    input_feav = Variable(self.input_fea)
                    input_semv = Variable(self.input_sem)

                    # loss of real data
                    criticD_real = self.netD(input_feav, input_semv)
                    criticD_real = criticD_real.mean()
                    criticD_real.backward(mone)
                    # loss of generated data
                    self.noise.normal_(0, 1)
                    noisev = Variable(self.noise)
                    fake = self.netG(noisev, input_semv)   # generate samples
                    # detach(): return a new variable, do not compute gradient for it
                    criticD_fake = self.netD(fake.detach(), input_semv)
                    criticD_fake = criticD_fake.mean()
                    criticD_fake.backward(one)

                    # loss with Lipschitz constraint
                    gradient_penalty = self.calc_gradient_penalty(self.input_fea, fake.data, self.input_sem)
                    gradient_penalty.backward()

                    # Wasserstein_D = criticD_real - criticD_fake
                    # Final Loss of Discriminator
                    D_cost = criticD_fake - criticD_real + gradient_penalty
                    self.optimizerD.step()

                for p in self.netD.parameters():  # reset requires_grad
                    p.requires_grad = False  # avoid computation
                # GENERATOR
                self.netG.zero_grad()
                input_semv = Variable(self.input_sem)
                self.noise.normal_(0, 1)
                noisev = Variable(self.noise)
                fake = self.netG(noisev, input_semv)
                criticG_fake = self.netD(fake, input_semv)
                criticG_fake = criticG_fake.mean()
                G_cost = -criticG_fake
                # classification loss
                c_errG = self.cls_criterion(self.pretrain_cls.model(fake), Variable(self.input_label))

                errG = G_cost + args.cls_weight * c_errG

                errG.backward()
                self.optimizerG.step()

            print('EP[%d/%d]******************************************************' % (epoch, args.epoch))

            # evaluate the model, set G to evaluation mode
            self.netG.eval()
            # train_X: input features (of unseen or seen) for training classifier2 in testing stage
            # train_Y: training labels
            # Generalized zero-shot learning
            if args.gzsl:
                syn_feature, syn_label = self.generate_syn_feature(args.syn_num)
                if args.dataset == 'AwA2':
                    train_X = torch.cat((self.data.train_seen_feature, syn_feature), 0)
                    train_Y = torch.cat((self.data.train_seen_label, syn_label), 0)
                    classes = torch.cat((self.data.seenclasses, self.data.unseenclasses), 0)
                    nclass = classes.size(0)
                    classifier_cls.CLASSIFIER(args, train_X, util.map_label(train_Y, classes), self.data, nclass, args.cuda,
                                                    args.cls_lr, 0.5, 50, 2 * args.syn_num, True)
                else:
                    train_X = torch.cat((self.data.train_seen_feature_sub, syn_feature), 0)
                    train_Y = torch.cat((self.data.train_seen_label_sub, syn_label), 0)
                    classes = torch.cat((self.data.seenclasses, self.data.unseenclasses), 0)
                    nclass = classes.size(0)
                    classifier_cls.CLASSIFIER(args, train_X, util.map_label(train_Y, classes), self.data, nclass, args.cuda,
                                                        args.cls_lr, 0.5, 50, 2 * args.batch_size, True)

            # Zero-shot learning
            else:
                # synthesize samples of unseen classes, for training classifier2 in testing stage
                syn_feature, syn_label = self.generate_syn_feature(args.syn_num)
                classifier_cls.CLASSIFIER(args, syn_feature, util.map_label(syn_label, self.data.unseenclasses), self.data,
                                                 self.data.unseenclasses.size(0), args.cuda, args.cls_lr, 0.5, 50, 10*args.syn_num, False, args.ratio, epoch)

            self.netG.train()
            # sys.stdout.flush()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    Data loading
    '''

    parser.add_argument('--data_dir', default='../../ZSL_data', help='path to save dataset')

    # parser.add_argument('--dataset', default='AwA2', help='for awa')
    parser.add_argument('--dataset', default='ImNet_A', help='for imagenet')

    parser.add_argument('--semantic_type', default='', type=str, help='the type of class embedding to input')
    parser.add_argument('--noise_size', type=int, default=100, help='size of noise vectors')

    '''
    Generator and Discriminator Parameter
    '''
    parser.add_argument('--NGH', default=4096, help='size of the hidden units in generator')
    parser.add_argument('--NDH', default=4096, help='size of the hidden units in discriminator')
    parser.add_argument('--critic_iter', default=5, help='critic iteration of discriminator, default=5, following WGAN-GP setting')
    parser.add_argument('--GP_weight', type=float, default=10, help='gradient penalty regularizer, default=10, the completion of Lipschitz Constraint in WGAN-GP')
    parser.add_argument('--cls_weight', default=0.01, help='loss weight for the supervised classification loss')
    parser.add_argument('--syn_num', default=300, type=int, help='number of features generating for each unseen class; awa_default = 300')

    '''
    Training Parameter
    '''
    parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--cuda', default=True, help='')
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument('--manual_seed', default=9416, type=int, help='')  #
    parser.add_argument('--batch_size', default=4096, type=int, help='')
    parser.add_argument('--epoch', default=100, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate to train GAN')
    parser.add_argument('--cls_lr', default=0.001, help='after generating unseen features, the learning rate for training softmax classifier')
    parser.add_argument('--ratio', default=0.1, help='ratio of easy samples')
    parser.add_argument('--beta', default=0.5, help='beta for adam, default=0.5')



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




    print(util.GetNowTime())
    print('Begin run!!!')

    run = Runner(args)
    run.train()

    print('End run!!!')
    print(util.GetNowTime())


