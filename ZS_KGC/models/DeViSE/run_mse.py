import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from collections import deque
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json



from data_loader import load_train_data
from networks import Extractor, DeViSE, weights_init
from functions import *



class Runner:
    def __init__(self, args):
        self.args = args

        print('############## LOADING .... ###################')

        self.train_tasks = json.load(open(os.path.join(args.data_path, 'datasplit', 'train_tasks.json')))
        self.rel2id = json.load(open(os.path.join(args.data_path, 'relation2ids')))

        self.rela_matrix = load_semantic_embed(args.data_path, args.dataset, args.semantic_type)
        self.input_dims = self.rela_matrix.shape[1]

        print('##LOADING ENTITY##')
        self.ent2id = json.load(open(os.path.join(args.data_path, 'entity2id')))
        num_ents = len(self.ent2id.keys())

        print('##LOADING CANDIDATES ENTITIES##')
        self.rel2candidates = json.load(open(os.path.join(args.data_path, 'rel2candidates_all.json')))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(os.path.join(args.data_path, 'e1rel_e2_all.json')))


        self.rela2label = dict()
        rela_sorted = sorted(list(self.train_tasks.keys()))
        for i, rela in enumerate(rela_sorted):
            self.rela2label[rela] = int(i)

        print('##LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        self.symbol2id, self.symbol2vec = self.read_embed()

        num_symbols = len(self.symbol2id.keys()) - 1  #
        print("num symbols:", num_symbols)
        pad_id = num_symbols

        # Pretraining step to obtain reasonable real data embeddings, load already pre-trained
        print('Load Pretrained Feature Encoder!')
        feature_encoder = Extractor(args.embed_dim, num_symbols, embed=self.symbol2vec).cuda()
        feature_encoder.apply(weights_init)
        model_path = os.path.join(args.data_path, 'FE_models_trained', args.embed_model + '_Extractor')
        feature_encoder.load_state_dict(torch.load(model_path))
        self.feature_encoder = feature_encoder
        self.feature_encoder.eval()


        print('##BUILDING CONNECTION MATRIX')
        self.degrees, self.connections, self.e1_degrees = self.build_connection(num_ents, pad_id, self.symbol2id, self.ent2id, max_=args.max_neighbor)

        print('################ DEFINE ZSL MODEL AND LOSS FUNCTION ... ##############')
        self.zsl_model = DeViSE(self.input_dims, args.hidden_dims, args.output_dims, args.p).cuda()
        self.loss_fn = nn.MSELoss()
        self.optimizer_tag = optim.Adam(self.zsl_model.parameters(), lr=args.lr, weight_decay=args.wds)
        print('using {} as criterion'.format(self.loss_fn))







    def read_embed(self):
        symbol_id = json.load(open(os.path.join(self.args.data_path, 'Embed_used', args.embed_model + '2id')))
        embeddings = np.load(os.path.join(self.args.data_path, 'Embed_used', args.embed_model + '.npz'))['arr_0']
        symbol2id = symbol_id
        symbol2vec = embeddings
        return symbol2id, symbol2vec

    #  build neighbor connection
    def build_connection(self, num_ents, pad_id, symbol2id, ent2id, max_=100):

        connections = (np.ones((num_ents, max_, 2)) * pad_id).astype(int)
        e1_rele2 = defaultdict(list)
        e1_degrees = defaultdict(int)
        # rel_list = list()
        with open(os.path.join(self.args.data_path, 'path_graph')) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                e1_rele2[e1].append((symbol2id[rel], symbol2id[e2]))
                e1_rele2[e2].append((symbol2id[rel], symbol2id[e1]))

        # print("path graph relations:", len(set(rel_list)))
        degrees = {}
        for ent, id_ in ent2id.items():
            neighbors = e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                connections[id_, idx, 0] = _[0]
                connections[id_, idx, 1] = _[1]


        return degrees, connections, e1_degrees

    def get_meta(self, left, right, connections, e1_degrees):
        left_connections = Variable(
            torch.LongTensor(np.stack([connections[_, :, :] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([e1_degrees[_] for _ in left])).cuda()

        right_connections = Variable(
            torch.LongTensor(np.stack([connections[_, :, :] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([e1_degrees[_] for _ in right])).cuda()

        return (left_connections, left_degrees, right_connections, right_degrees)



    def train(self):


        print('\n############ START TRAINING ... ############')
        losses = deque([], args.loss_every)


        train_data = load_train_data(self.args, self.train_tasks, self.symbol2id, self.ent2id,
                                           self.e1rel_e2, self.rel2id, self.rela2label, self.rela_matrix)



        for epoch in range(1, (args.train_times+1)):
            self.zsl_model.train()


            rel_sem, rel_sem_neg, query, query_left, query_right, false, false_left, false_right, labels = train_data.__next__()

            rel_sem = Variable(torch.FloatTensor(rel_sem)).cuda()


            # encoding (h, t) pair
            query_meta = self.get_meta(query_left, query_right, self.connections, self.e1_degrees)
            query = Variable(torch.LongTensor(query)).cuda()

            entity_pair_vector, _ = self.feature_encoder(query, query, query_meta, query_meta)

            self.zsl_model.zero_grad()

            rel_sem_mapped = self.zsl_model(rel_sem)
            loss = self.loss_fn(entity_pair_vector, rel_sem_mapped)


            loss.backward()
            self.optimizer_tag.step()


            losses.append(loss.item())



            if epoch % args.loss_every == 0:

                print("Epoch: %d, loss: %.3f" % (epoch, np.mean(losses)))


            if epoch >= 1000 and epoch % args.eval_every == 0:
                self.test(epoch)
                # self.save_model()

    def test(self, epoch=0):
        self.zsl_model.eval()

        # test_candidates = json.load(open(os.path.join(self.args.data_path, "test_candidates_sub_10.json")))
        test_candidates = json.load(open(os.path.join(self.args.data_path, "test_candidates.json")))

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []

        for query_ in sorted(test_candidates.keys()):


            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            rel_sem = self.rela_matrix[self.rel2id[query_]]
            rel_sem = np.expand_dims(rel_sem, axis=0)
            rel_sem = Variable(torch.FloatTensor(rel_sem)).cuda()

            rel_sem_mapped = self.zsl_model(rel_sem)
            rel_sem_mapped.detach()
            rel_sem_mapped = rel_sem_mapped.data.cpu().numpy()

            for e1_rel, tail_candidates in test_candidates[query_].items():
                if args.dataset == "NELL":
                    head, rela, _ = e1_rel.split('\t')
                elif args.dataset == "Wiki":
                    head, rela = e1_rel.split('\t')
                else:
                    print('The Dataset is not supported, please check')

                true = tail_candidates[0]
                query_pairs = []
                if head not in self.symbol2id or true not in self.symbol2id:
                    continue
                query_pairs.append([self.symbol2id[head], self.symbol2id[true]])

                query_left = []
                query_right = []
                query_left.append(self.ent2id[head])
                query_right.append(self.ent2id[true])

                for tail in tail_candidates[1:]:
                    if tail not in self.symbol2id:
                        continue
                    query_pairs.append([self.symbol2id[head], self.symbol2id[tail]])
                    query_left.append(self.ent2id[head])
                    query_right.append(self.ent2id[tail])

                query = Variable(torch.LongTensor(query_pairs)).cuda()

                query_meta = self.get_meta(query_left, query_right, self.connections, self.e1_degrees)
                candidate_vecs, _ = self.feature_encoder(query, query, query_meta, query_meta)
                candidate_vecs.detach()
                candidate_vecs = candidate_vecs.data.cpu().numpy()
                scores = cosine_similarity(candidate_vecs, rel_sem_mapped)
                scores = np.squeeze(scores, axis=1)

                assert scores.shape == (len(query_pairs),)

                sort = list(np.argsort(scores))[::-1]  # ascending -> descending
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



        print('\n############   ' + 'TEST' + ' ' + str(epoch) + '    #############')
        print('HITS10: {:.3f}, HITS5: {:.3f}, HITS1: {:.3f}, MAP: {:.3f}'.format(np.mean(hits10),
                                                                                 np.mean(hits5),
                                                                                 np.mean(hits1),
                                                                                 np.mean(mrr)))
        print('###################################')






if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../../ZSL_data')
    parser.add_argument('--dataset', default='NELL')

    parser.add_argument("--embed_model", default='TransE', type=str)
    parser.add_argument("--max_neighbor", default=50, type=int, help='neighbor number of each entity')
    parser.add_argument("--embed_dim", default=100, type=int, help='dimension of triple embedding')
    parser.add_argument("--ep_dim", default=200, type=int, help='dimension of entity pair embedding')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--batch_rela_num", default=4, type=int)
    parser.add_argument("--train_times", default=7000, type=int)

    # parser.add_argument('--output_dims', default=200, type=int, help='the dimension of instance (entity pair) feature')
    parser.add_argument("--loss_every", default=50, type=int)
    parser.add_argument("--eval_every", default=500, type=int)

    parser.add_argument('--p', default=0.5, help='dropout', type=float)
    parser.add_argument('--lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--wds', default=1e-5, help='', type=float)
    parser.add_argument('--manual_seed', default=12345, help='', type=int)
    parser.add_argument("--semantic_type", default='rdfs', help='{options: text, rdfs, rdfs_hie, rdfs_cons, rdfs_text}')
    parser.add_argument('--hidden_dims', type=int, default=300, help='the dimension of relation semantic embedding')

    parser.add_argument('--gpu', default=1, help='gpu id', type=int)

    args = parser.parse_args()

    args.data_path = os.path.join(args.data_dir, args.dataset)
    args.output_dims = args.ep_dim




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
    else:
        print("GPU is not available!")





    run = Runner(args)
    run.train()
