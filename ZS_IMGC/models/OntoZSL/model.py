import torch.nn as nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_CRITIC(nn.Module):
    def __init__(self, args):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(args.feat_dim + args.sem_dim, args.NDH)
        # self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(args.NDH, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, sem):
        h = torch.cat((x, sem), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_G(nn.Module):
    def __init__(self, args):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(args.sem_dim + args.noise_size, args.NGH)
        self.fc2 = nn.Linear(args.NGH, args.feat_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, sem):
        h = torch.cat((noise, sem), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h





