import keras

from compgcn_conv import CompGCNConv
from helper import *
from data_generator import *
from models import CompGCNTransE, CompGCNDistMult, CompGCNConvE
from metrics import MRR, Hits


class Runner:

    def load_dataset(self):
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)]})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}'.format(split)].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}'.format(split)].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})
        self.triples = dict(self.triples)
        self.load_generator()

    def load_generator(self):
        self.data_iter = {
            'train': DataGenerator(self.triples['train'], self.p, batch_size=self.p.batch_size),
            'valid': DataGenerator(self.triples['valid'], self.p, batch_size=self.p.batch_size),
            'test': DataGenerator(self.triples['test'], self.p, batch_size=self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = tf.transpose(tf.constant(edge_index, dtype='int64'))
        edge_type = tf.constant(edge_type, dtype='int64')
        return edge_index, edge_type

    def add_model(self, score_func):
        model_name = '{}'.format(score_func)
        if model_name.lower() == 'transe':
            model = CompGCNTransE
        elif model_name.lower() == 'distmult':
            model = CompGCNDistMult
        elif model_name.lower() == 'conve':
            model = CompGCNConvE
        else:
            raise NotImplementedError
        return model

    def __init__(self, params):

        best_val_mrr, best_val, best_epoch, val_mrr = 0., {}, 0, 0.

        self.p = params

        self.load_dataset()
        self.mrr = MRR()
        self.hits_1 = Hits(1)
        self.hits_3 = Hits(3)
        self.hits_10 = Hits(10)
        model = self.add_model(self.p.score_func)
        model = model(self.edge_index, self.edge_type, self.p.num_rel,
                              (self.mrr, self.hits_1, self.hits_3, self.hits_10), params=self.p)
        kill_cnt = 0

        for epoch in range(self.p.epochs):
            losses = []
            for step, batch in enumerate(iter(self.data_iter['train'])):
                x, y = batch
                train_loss = model.train_step(x, y)
                losses.append(train_loss)
                if step % 100 == 0:
                    print('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}'.format(epoch, step, np.mean(losses),
                                                                                  best_val_mrr))
            print('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, np.mean(losses)))
            for step, batch in enumerate(iter(self.data_iter['valid'])):
                x, y = batch
                model.test_step(x, y)
                if step % 100 == 0:
                    print('[{}, Step {}]'.format('Valid', step))
            valid_mrr = self.mrr.result()
            valid_hits1 = self.hits_1.result()
            valid_hits3 = self.hits_3.result()
            valid_hits10 = self.hits_10.result()
            self.mrr.reset_state()
            self.hits_1.reset_state()
            self.hits_3.reset_state()
            self.hits_10.reset_state()
            tf.print("Valid MRR:", valid_mrr," Valid Hits 1:",valid_hits1, "Valid Hits 3:", valid_hits3, "Valid Hits 10:", valid_hits10)
            if valid_mrr > best_val_mrr:
                best_weights = model.get_weights()
                best_val_mrr = valid_mrr
                best_epoch = epoch
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt > 25:
                    print("Early Stopping!!")
                    break
            print('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, best_val_mrr))
        print('Loading best weights, Evaluating on Test data')
        model.set_weights(best_weights)
        for step, batch in enumerate(iter(self.data_iter['test'])):
            x, y = batch
            model.test_step(x, y)
            if step % 100 == 0:
                print('[{}, Step {}]'.format('Test', step))
        test_mrr = self.mrr.result()
        test_hits1 = self.hits_1.result()
        test_hits3 = self.hits_3.result()
        test_hits10 = self.hits_10.result()
        self.hits_1.reset_state()
        self.hits_3.reset_state()
        self.hits_10.reset_state()
        self.mrr.reset_state()
        tf.print("Test MRR:",test_mrr, "Test Hits 1:", test_hits1, "Test Hits 3:", test_hits3, "Test Hits 10:", test_hits10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-score_func', dest='score_func', default='transe', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='corr', help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epochs', dest='epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('-init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', dest='embed_dim', default=None, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-seed', dest='seed', default=0, type=int, help='Seed for randomization')

    # ConvE specific hyperparameters
    parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')
    args = parser.parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_gpu(args.gpu)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    Runner(args)
