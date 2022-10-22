import torch.nn as nn

class devise(nn.Module):
    def __init__(self, input_dims, output_dims, p):
        super(devise, self).__init__()
        self.model = nn.Sequential(nn.BatchNorm1d(input_dims),
                         nn.Dropout(p),
                         nn.Linear(in_features=input_dims, out_features=2048, bias=True),
                         nn.ReLU(),
                         nn.BatchNorm1d(2048),
                         nn.Dropout(p),
                         nn.Linear(in_features=2048, out_features=output_dims, bias=True))
    def forward(self, x):
        x = self.model(x)
        return x