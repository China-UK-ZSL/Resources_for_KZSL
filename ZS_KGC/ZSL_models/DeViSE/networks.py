import logging
import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class SupportEncoder(nn.Module):
    """docstring for SupportEncoder"""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))


        return self.layer_norm(output + residual)

class Extractor(nn.Module):
    """
    Matching metric based on KB Embeddings
    """

    def __init__(self, embed_dim, num_symbols, embed=None):
        super(Extractor, self).__init__()
        self.embed_dim = int(embed_dim)
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(self.embed_dim, int(self.embed_dim / 2))
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.fc1 = nn.Linear(self.embed_dim, int(self.embed_dim / 2))
        self.fc2 = nn.Linear(self.embed_dim, int(self.embed_dim / 2))

        self.dropout = nn.Dropout(0.2)
        self.dropout_e = nn.Dropout(0.2)

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout=0.2)
        # self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        entities = connections[:, :, 1].squeeze(-1)
        ent_embeds = self.dropout(self.symbol_emb(entities))  # (batch, 50, embed_dim)
        concat_embeds = ent_embeds

        out = self.gcn_w(concat_embeds)
        out = torch.sum(out, dim=1)  # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    def entity_encoder(self, entity1, entity2):
        entity1 = self.dropout_e(entity1)
        entity2 = self.dropout_e(entity2)
        entity1 = self.fc1(entity1)
        entity2 = self.fc2(entity2)
        entity = torch.cat((entity1, entity2), dim=-1)
        return entity.tanh()  # (batch, embed_dim)

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_e1 = self.symbol_emb(query[:, 0])  # (batch, embed_dim)
        query_e2 = self.symbol_emb(query[:, 1])  # (batch, embed_dim)
        query_e = self.entity_encoder(query_e1, query_e2)

        support_e1 = self.symbol_emb(support[:, 0])  # (batch, embed_dim)
        support_e2 = self.symbol_emb(support[:, 1])  # (batch, embed_dim)
        support_e = self.entity_encoder(support_e1, support_e2)

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_e, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat((support_left, support_e, support_right), dim=-1)  # tanh

        support = support_neighbor
        query = query_neighbor

        support_g = self.support_encoder(support)  # 1 * 100
        query_g = self.support_encoder(query)



        support_g = torch.mean(support_g, dim=0, keepdim=True)

        # cosine similarity
        matching_scores = torch.matmul(query_g, support_g.t()).squeeze()

        return query_g, matching_scores



class DeViSE(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, p):
        super(DeViSE, self).__init__()
        self.model = nn.Sequential(nn.BatchNorm1d(input_dims),
                         nn.Dropout(p),
                         nn.Linear(in_features=input_dims, out_features=hidden_dims, bias=True),
                         nn.ReLU(),
                         nn.BatchNorm1d(hidden_dims),
                         nn.Dropout(p),
                         nn.Linear(in_features=hidden_dims, out_features=output_dims, bias=True))
    def forward(self, x):
        x = self.model(x)
        return x