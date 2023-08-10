import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False
from torch_geometric.loader import NeighborLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import f1_score
import wandb

import utils
import config_load
from model import *
from loss import UnionLoss
from data_preprocess import get_data, CancerDataset

def arg_parse():
    parser = argparse.ArgumentParser(description="Train GATRes arguments.")
    parser.add_argument('-a', "--ablation", dest="ablation", action="store_true")
    parser.add_argument('-cv, "--cross_validation', dest="cv",
                        help="use cross validation", action="store_true")
    parser.add_argument('-cl, "--change_loss', dest="change_loss", action="store_true")
    parser.add_argument('--data_dir', type=str, default='data/Lung_Cancer_Matrix',)
    parser.add_argument('-d', "--debug", dest="debug", action="store_true")
    parser.add_argument('-e', "--encode", dest="encode", action="store_true")
    parser.add_argument('-f', "--fold", dest='fold', help="fold", default=4, type=int)
    parser.add_argument('-g', '--gpu', dest='gpu', default=3)
    parser.add_argument('-m', "--model", dest='model', default='Multi_GTN')
    parser.add_argument('-n', '--n2v', dest='n2v', action='store_true')
    parser.add_argument('-op', '--overlap', action='store_true', default=False,)
    parser.add_argument('-p', "--predict", dest='pred', help="predict all nodes", action="store_true")
    parser.add_argument('--ppi', dest='ppi', default=None, nargs='*')
    parser.add_argument('-sf', "--start_fold", dest='start_fold', default=0, type=int)
    parser.add_argument('-sh', '--select_hyperparams', dest='sh', action='store_true')
    parser.add_argument('-t', "--train", dest="train", action="store_true")
    return parser.parse_args()

class MultiPPI_Encoder():
    def __init__(self, args) -> None:
        assert args['mode'] in ['train', 'pred', 'encode']

        self.args = args
        self.mode = args['mode']
        self.dataset = get_data(args, args['stable'])
        self.num_node_features = self.dataset.num_node_features
        self.log_name = args['log_file'].split('/')[-1].split('.')[0]
        ppi_dict = {
            'CPDB': 1,
            'IRef': 2,
            'Multinet': 3,
            'PCNet': 4,
            'STRING': 5,
        }
        if len(args['PPI']) == 1:
            iterator = iter(self.dataset)
            for _ in range(ppi_dict[args['PPI'][0]]):
                mid = next(iterator)
            self.ori_dataset = self.dataset
            self.dataset = [mid]

        self.drop_idx = []
        self.loss_func = torch.nn.BCELoss() if args['loss'] == 'BCE' \
            else UnionLoss(
            pos_lambda=args['pos_lambda'],
            neg_lambda=args['neg_lambda'],
            func=args['loss'],
            start_epoch=args['start_epoch'],)
        
        if args['model'] == 'MTGCN':
            self.loss_func = torch.nn.BCEWithLogitsLoss()
        elif args['model'] != 'Multi_GTN':
            self.loss_func = torch.nn.BCELoss()
        
        
    @property
    def best_model(self,):
        model_dir = os.path.join(self.args['out_dir'], self.log_name,)
        models = os.listdir(model_dir)
        best_f1 = 0.
        for model in models:
            f1 = float(model.split('_')[1])
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_fold = int(model.split('_')[0])
        print(f'Best fold is {best_fold} with f1 {best_f1}.')
        return best_model
        

    def drop_samples(self, fold, sample_neg=0., sample_pos=1., num_samples=0):
        # 必须和恢复成对使用
        assert len(self.drop_idx) == 0
        self.args["drop_neg"] = 0. if self.args['mode'] == 'pred' else self.args["drop_neg"]
        print(f"Drop {self.args['drop_neg']} of negative train samples in fold {fold}")

        if self.args["sample_rate"] != 1: # 这里还没有实现想要的功能
            print(f"Drop {1 - self.args['sample_rate']} of train samples")

        if sample_neg == 1 and sample_pos == 1:
            return []
        drop_neg = 1 - sample_neg
        drop_pos = 1 - sample_pos
        splitted_idx = self.dataset.get_idx_split(fold) if len(self.args['PPI']) == 5 else self.ori_dataset.get_idx_split(fold)
        train_idx = splitted_idx['train']
        drop_neg_idx, drop_pos_idx = [], []
        for i in train_idx:
            if self.dataset[0].y[i][0]:
                drop_neg_idx.append(i.item())
            if self.dataset[0].y[i][1]:
                drop_pos_idx.append(i.item())
        num_neg_samples = len(drop_neg_idx)
        num_pos_samples = len(drop_pos_idx)
        random.seed(self.args['random_seed'])
        if num_samples:
            num_neg = int(num_samples * num_neg_samples /
                        (num_neg_samples + num_pos_samples))
            num_pos = num_samples - num_neg
            drop_neg_idx = random.sample(drop_neg_idx, num_neg_samples - num_neg)
            drop_pos_idx = random.sample(drop_pos_idx, num_pos_samples - num_pos)
        else:
            drop_neg_idx = random.sample(
                drop_neg_idx, int(num_neg_samples*drop_neg))
            drop_pos_idx = random.sample(
                drop_pos_idx, int(num_pos_samples*drop_pos))
        drop_idx = self.drop_idx = drop_neg_idx + drop_pos_idx
        print(
            f"Negatives: {num_neg_samples - len(drop_neg_idx)}, Positives: {num_pos_samples - len(drop_pos_idx)}")
        
        for i in range(len(self.dataset)):
            self.dataset[i].train_mask[drop_idx, fold] = False
    

    def recover_drop(self, fold):
        for i in range(len(self.dataset)):
            self.dataset[i].train_mask[self.drop_idx, fold] = True
            self.drop_idx = []
        
        
    def load_data(self, fold):
        self.train_loader_list, self.valid_loader_list, self.test_loader_list, self.unknown_loader_list = [], [], [], []

        for data in self.dataset:
            data.contiguous()
            if self.mode == 'pred':
                train_mask = data.train_mask[:, fold] + data.valid_mask[:, fold] + data.test_mask[:, fold]
            
            elif self.mode == 'encode':
                train_mask = data.train_mask[:, fold] + data.valid_mask[:, fold] + data.test_mask[:, fold] + data.unlabeled_mask

            elif self.mode == 'train':
                train_mask = data.train_mask[:, fold]

                self.valid_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    directed=False,
                    input_nodes=data.valid_mask[:, fold]))
                
                self.test_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    directed=False,
                    input_nodes=data.test_mask[:, fold],))
                
            self.train_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    directed=False,
                    input_nodes=train_mask))
            
            self.unknown_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    directed=False,
                    input_nodes=data.unlabeled_mask))


    def init_model(self):
        args = self.args
        if 'Multi' in args['model']:
            model = self.model = Multi_GTN(
                gnn=args['model'],
                in_channels=self.num_node_features, 
                hidden_channels=args['hidden_channels'], 
                heads=args['heads'], 
                drop_rate=args['drop_rate'],
                attn_drop_rate=args['attn_drop'], 
                edge_dim=self.dataset[0].edge_dim,
                num_ppi=len(self.dataset),
                pooling=args['pooling'],
                residual=args['residual'],
                learnable_weight=args['learnable_factor']).to(args['gpu'])

        elif 'GCN' == args['model']:
            model = self.model = GCN(
                in_channels=self.num_node_features, 
                hidden_channels=args["hidden_channels"],
                drop_rate=args["drop_rate"],
                residual=args["residual"]).to(args['gpu'])
            
        elif 'GAT' == args['model']:
            model = self.model = GAT(
                in_channels=self.num_node_features,
                hidden_channels=args["hidden_channels"],
                heads=args["heads"],
                drop_rate=args["drop_rate"],  
                edge_dim=self.dataset[0].edge_dim,
                residual=args["residual"]).to(args['gpu'])
            
        elif 'EMOGI' == args['model']:
            model = self.model = EMOGI(
                in_channels=self.num_node_features,
                drop_rate=0.5,
                hidden_dims=[300, 100],
                residual=args['residual']).to(args['gpu'])
            
        elif 'MTGCN' == args['model']:
            model = self.model = MTGCN(
                in_channels=self.num_node_features, 
                hidden_dims=[300, 100],
                residual=args["residual"]).to(args['gpu'])
            
        
        if args["model"] == "EMOGI":
            opt_list = [dict(params=model.convs[0].parameters(), weight_decay=0.005)] + \
                [dict(params=model.convs[i].parameters(), weight_decay=0) for i in range(1, len(model.convs))]
            self.optimizer = t.optim.Adam(opt_list, lr=args['lr'])

        else:
            self.optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=args['weight_decay'], lr=args['lr'])

        num_train_steps = args['num_epochs']
        # sum(self.dataset[0].train_mask[:, 0]) / args['batch_size'] * args['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.2 * num_train_steps,
            num_training_steps=num_train_steps)
        
    
    def train_epoch(self, epoch):
        self.model.train()
        tot_loss, steps = 0, 0
        for data_tuple in zip(*self.train_loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            steps += 1
            self.optimizer.zero_grad()
            size = data_tuple[0].batch_size

            if isinstance(self.model, Multi_GTN):
                out, x_list, ppi_weight = self.model(data_tuple)
            else:
                out = self.model(data_tuple)
            if isinstance(self.model, MTGCN):
                out, rl, c1, c2 = out[0][:size], out[1], out[2], out[3]

            true_lab = data_tuple[0].y[:size, 1]
            out = out.view(-1)
            if isinstance(self.model, Multi_GTN):
                loss = self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']), epoch)
            else:
                loss = self.loss_func(out, true_lab.float())
            if isinstance(self.model, MTGCN):
                loss = loss / (c1 ** 2) + rl / (c2 ** 2) + 2 * torch.log(c1 * c2)

            del out, true_lab
            loss.backward()
            tot_loss = tot_loss + loss.item()
            self.optimizer.step()
        self.scheduler.step()
        tot_loss = tot_loss / steps

        self.model.eval()
        y_true = np.array([])
        y_score = np.array([])
        y_pred = np.array([])

        for data_tuple in zip(*self.train_loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            size = data_tuple[0].batch_size
            with torch.no_grad():
                if isinstance(self.model, Multi_GTN):
                    out, _, _ = self.model(data_tuple)
                else:
                    out = self.model(data_tuple)
                if isinstance(self.model, MTGCN):
                    out, _, _, _ = out[0][:size], out[1], out[2], out[3]

            true_lab = data_tuple[0].y[:size][:, 1]
            out = out.view(-1)
            pred_lab = np.zeros(size)
            if size == 1:
                pred_lab[0] = 1 if out[0] > 0.5 else 0
            else:
                pred_lab[out.cpu() > 0.5] = 1
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
            y_pred = np.append(y_pred, pred_lab, axis=0)
            y_true = np.append(y_true, true_lab.cpu().detach().numpy())

        train_acc, _, auprc, train_f1, auc = utils.calculate_metrics(y_true, y_pred, y_score)
        print(f"Train Loss: {tot_loss:.6f}, F1: {train_f1:.4f}, ACC: {train_acc:.4f}, "\
              f"AUROC: {auc:.4f}, AUPRC: {auprc:.4f}")
        
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        valid_loss = 0
        steps = 0
        ppi_weight = None
        for data_tuple in zip(*self.valid_loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            steps = steps + 1
            size = data_tuple[0].batch_size
            with torch.no_grad():
                if isinstance(self.model, Multi_GTN):
                    out, x_list, ppi_weight = self.model(data_tuple)
                else:
                    out = self.model(data_tuple)
                if isinstance(self.model, MTGCN):
                    out, rl, c1, c2 = out[0][:size], out[1], out[2], out[3]

            true_lab = data_tuple[0].y[:size][:, 1]
            out = out.view(-1)
            pred_lab = np.zeros(size)
            if size == 1:
                pred_lab[0] = 1 if out[0] > 0.5 else 0
            else:
                pred_lab[out.cpu() > 0.5] = 1
            y_pred = np.append(y_pred, pred_lab)
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
            y_true = np.append(y_true, true_lab.cpu().detach().numpy())
            if isinstance(self.model, Multi_GTN):
                valid_loss += self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']), epoch).item()
            else:
                loss = self.loss_func(out, true_lab.float()).item()
            if isinstance(self.model, MTGCN):
                valid_loss = valid_loss + loss / (c1 ** 2) + rl / (c2 ** 2) + 2 * torch.log(c1 * c2)
                valid_loss = valid_loss.item()
            else:
                valid_loss += loss
    

        valid_loss = valid_loss / steps
        acc, cf_matrix, auprc, f1, auc = utils.calculate_metrics(y_true, y_pred, y_score)

        return tot_loss, valid_loss, f1, acc, auc, auprc, cf_matrix, train_f1, train_acc, ppi_weight



    def grid_search_SVM(self,):
        train_mask = self.dataset[0].train_mask[:, 0]
        valid_mask = self.dataset[0].valid_mask[:, 0]

        x_train = self.dataset[0].x[train_mask + valid_mask]
        y_train = self.dataset[0].y[train_mask + valid_mask][:, 1]

        param_grid = {
            'C': [1/4, 1/2, 1, 2],
            'kernel': ['rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
        self.model = SVM()
        grid_search = GridSearchCV(self.model.svm, param_grid, scoring='f1', cv=10, verbose=2)
        for _ in tqdm(range(1)):
            grid_search.fit(x_train, y_train)
        
        return grid_search.best_params_
    

    def train_SVM(self):
        model = SVM()
        best_params = self.grid_search_SVM()
        with open(self.args['log_file'], 'a') as f:
            print("Best params:", best_params, file=f, flush=True)

        train_result = []
        for fold in range(self.args['cv_folds']):
            test_mask = self.dataset[0].test_mask[:, fold]
            x_test = self.dataset[0].x[test_mask].numpy()
            y_test = self.dataset[0].y[test_mask][:, 1].numpy()
            model = SVM(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
            train_mask = self.dataset[0].train_mask[:, fold]
            x_train = self.dataset[0].x[train_mask]
            y_train = self.dataset[0].y[train_mask][:, 1]

            model.svm.fit(x_train, y_train)
            y_pred = model.svm.predict(x_test)
            acc = model.svm.score(x_test, y_test)
            f1 = f1_score(y_test, y_pred)
            train_result.append([f1, acc])

        with open(self.args['log_file'], 'a') as f:
            print("SVM: ", np.mean(train_result, axis=0), np.std(train_result, axis=0), file=f, flush=True)
            

    def train_single_fold(self, fold, print_args, ckpt=None,):
        self.init_model()
        if ckpt:
            self.model.load_state_dict(torch.load(ckpt)['state_dict'])
        self.drop_samples(fold, (1 - self.args['drop_neg']))
        self.load_data(fold)
        
        log_path = os.path.dirname(self.args['log_file'])
        print(f"Log Path is /{log_path}/{self.log_name}")
        Id = utils.generate_random_name()
        if print_args:
            utils.print_config(self.args['log_file'], self.args)
            with open(self.args['log_file'], 'a') as f:
                print(f"Train: {self.dataset[0].train_mask[:, fold].sum().item()} "\
                      f"Valid: {self.dataset[0].valid_mask[:, fold].sum().item()} "\
                      f"Test: {self.dataset[0].test_mask[:, fold].sum().item()} ",
                      file=f)
                print(self.model, file=f, flush=True)
        with open(self.args['log_file'], 'a') as f:
                print(f"Id:{Id}, Model: {self.args['model']}", file=f)
        if self.args['wandb']:
            wandb.init(project=self.args['project'],
                       config=self.args,
                       entity='starrybei',
                       mode='offline',
                       name=Id,
                       settings=wandb.Settings(start_method='fork'))
            wandb.watch(self.model, log="all")
        print("Start training")
        vmax_f1 = 0
        trigger_times = 0
        pos_lambda = self.args['pos_lambda']
        neg_lambda = self.args['neg_lambda']
        for epoch in range(self.args['num_epochs']):
            self.args['pos_lambda'] = 0. if epoch < 100 else pos_lambda
            self.args['neg_lambda'] = 0. if epoch < 100 else neg_lambda
            train_loss, valid_loss, f1, acc, auc, auprc, cf_matrix, train_f1, train_acc, ppi_weight = self.train_epoch(epoch)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch}, Valid loss: {valid_loss:.6f}, F1: {f1:.4f}, Acc: {acc:.4f}, "\
                    f"Auroc: {auc:.4f}, Auprc: {auprc:.4f}, TP: {cf_matrix[1, 1]}")
                if (not self.args['overlap']) and len(self.args['PPI']) == 5 and ppi_weight is not None:
                    print(f"CPDB: {ppi_weight[0].item():.4f}, IRef: {ppi_weight[1].item():.4f}, "\
                        f"Multinet: {ppi_weight[2].item():.4f}, PCNet: {ppi_weight[3].item():.4f}, STRING: {ppi_weight[4].item():.4f}")
            if self.args['wandb']:
                wandb.log({
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "Acc": acc,
                    "AUPRC": auprc,
                    "True_negative:": cf_matrix[0, 0],
                    "False_positive": cf_matrix[0, 1],
                    "False_negative": cf_matrix[1, 0],
                    "True_positive": cf_matrix[1, 1],
                    "auc": auc,
                    "f1": f1,
                    "train_f1": train_f1,
                    "train_acc": train_acc,})

            if epoch >= self.args["num_epochs"] // 5:
                if f1 < vmax_f1:
                    trigger_times += 1
                    if trigger_times == self.args["num_epochs"] // 5:
                        print("Early Stopping")
                        break
                else:
                    trigger_times = 0
                    vmax_f1 = f1
                    max_epoch = epoch
                    best_acc = acc
                    best_tp = cf_matrix[1, 1]
                    best_auc = auc
                    best_auprc = auprc
                    checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict()}
        if not os.path.exists(os.path.join(self.args['out_dir'], self.log_name)):
            os.mkdir(os.path.join(self.args['out_dir'], self.log_name))
        model_dir = os.path.join(self.args['out_dir'], self.log_name, f"{self.args['model']}_{fold}_{vmax_f1:.4f}_{best_acc:.4f}.pkl")
        torch.save(checkpoint, model_dir)
        with open(self.args['log_file'], 'a') as f:
            print("Fold {} epoch {}: F1:{:.4f}, ACC:{:.4f}, AUROC:{:.4f}, AUPRC:{:.4f}, TP:{:.1f}".format(
                fold, max_epoch, vmax_f1, best_acc, best_auc, best_auprc, best_tp), file=f, flush=True)
        self.recover_drop(fold)
        if self.args['wandb']:
            wandb.finish()
        return vmax_f1, model_dir


    def test(self, model_dir, pred, output=False):
        self.init_model()
        self.model.load_state_dict(torch.load(model_dir)['state_dict'])
        print(f"Loading model from {model_dir} ......")
        self.model.eval()

        y_true = np.array([]) if not pred else None
        y_pred = np.array([])
        y_score = np.array([])
        y_index = np.array([])

        if output is not False:
            self.load_data(output)
            loader_list = self.unknown_loader_list if pred else self.train_loader_list
        else:
            loader_list = self.unknown_loader_list if pred else self.test_loader_list

        for data_tuple in zip(*loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            size = data_tuple[0].batch_size
            with torch.no_grad():
                if isinstance(self.model, Multi_GTN):
                    out, _, _ = self.model(data_tuple)
                else:
                    out = self.model(data_tuple)
                if isinstance(self.model, MTGCN):
                    out, _, _, _ = out[0][:size], out[1], out[2], out[3]

            out = torch.squeeze(out)
            index = data_tuple[0].pos[:size]
            true_lab = data_tuple[0].y[:size][:, 1] if not pred else None
            pred_lab = np.zeros(size)
            if size == 1:
                pred_lab[0] = 1 if out[0] > 0.5 else 0
            else:
                pred_lab[out.cpu() > 0.5] = 1

            y_score = np.append(y_score, out.cpu().detach().numpy(), axis=0)
            y_pred = np.append(y_pred, pred_lab)
            y_index = np.append(y_index, index.cpu().detach().numpy())
            y_true = np.append(y_true, true_lab.cpu().detach().numpy()) if not pred else None

        if output is not False:
            return y_true, y_score, y_index

        acc, cf_matrix, auprc, f1, auc = utils.calculate_metrics(y_true, y_pred, y_score)
        tp = cf_matrix[1, 1]

        with open(self.args['log_file'], 'a') as f:
            print("Test F1:{:.4f}, ACC:{:.4f}, AUROC:{:.4f}, AUPRC:{:.4f}, TP:{:.1f}"
                .format(f1, acc, auc, auprc, tp), file=f, flush=True)
        return f1, acc, auc, auprc, cf_matrix


    def train(self, ):
        args = self.args
        ckpt = None
        test_auprc, test_auc, test_acc, test_f1, test_tp = [], [], [], [], []
        for j in range(args['repeat']):
            print_args = True if j == 0 else False
            if args['cv']:
                if args['start_fold'] > 0:
                    with open(args['log_file'], 'r') as f:
                        lines = f.readlines()
                        for l in range(-1 - (args['start_fold'] - 1) * 2, 0, 2):
                            if lines[l].startswith('Test'):
                                test_f1.append(float(lines[l].split(',')[0].split(':')[-1]))
                                test_acc.append(float(lines[l].split(',')[1].split(':')[-1]))
                                test_auc.append(float(lines[l].split(',')[2].split(':')[-1]))
                                test_auprc.append(float(lines[l].split(',')[3].split(':')[-1]))
                                test_tp.append(float(lines[l].split(',')[4].split(':')[-1]))
                            else: print("Error in reading log file")

                for i in range(args['start_fold'], args['cv_folds']):
                    print_args = True if i == 0 else False
                    args['fold'] = i
                    _, model_dir = self.train_single_fold(i, print_args, ckpt)
                    f1, acc, auc, auprc, cf_matrix = self.test(model_dir, pred=False)
                    test_auprc.append(auprc)
                    test_auc.append(auc)
                    test_acc.append(acc)
                    test_f1.append(f1)
                    test_tp.append(cf_matrix[1, 1])
            else:
                _, model_dir = self.train_single_fold(args['fold'], print_args, ckpt)

        if args['cv'] or args['repeat'] > 1:
            with open(args['log_file'], 'a') as f:
                print(f"Avg Test F1:{np.average(test_f1):.4f}±{np.std(test_f1):.4f}, "\
                    f"ACC:{np.average(test_acc):.4f}±{np.std(test_acc):.4f}, "\
                    f"AUROC:{np.average(test_auc):.4f}±{np.std(test_auc):.4f}, "\
                    f"AUPRC:{np.average(test_auprc):.4f}±{np.std(test_auprc):.4f}, "\
                    f"TP:{np.average(test_tp):.1f}±{np.average(test_tp):.1f}\n",
                    file=f, flush=True)
                
                
    def encode(self, out_dir):
        self.init_model()
        self.model.load_state_dict(torch.load(os.path.join(self.args['out_dir'], self.log_name, self.best_model))['state_dict'])
        self.load_data(fold=0)
        self.model.eval()

        all_x = np.array([])
        all_index = np.array([])

        for data_tuple in tqdm(zip(*self.train_loader_list), total=len(self.train_loader_list[0]), desc='Encoding'):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            size = data_tuple[0].batch_size
            with torch.no_grad():
                _, x_list, _ = self.model(data_tuple)
            x = t.cat(x_list, dim=1)
            index = data_tuple[0].pos[:size]

            if all_x.shape[0] == 0:
                all_x = x.cpu().detach().numpy()
            else:
                all_x = np.append(all_x, x.cpu().detach().numpy(), axis=0)
            all_index = np.append(all_index, index.cpu().detach().numpy())

        gene_list = utils.get_gene_list()
        gene_list['index'] = np.arange(gene_list.shape[0])
        embeddings = pd.DataFrame(data=all_x, columns=[i for i in range(all_x.shape[1])])
        embeddings['index'] = all_index
        embeddings = gene_list.merge(embeddings)
        embeddings = embeddings.drop('index', axis=1)
        embeddings.to_csv(out_dir, sep='\t', index=False)
        

def select_hyperparams(args, param_dict, log_file):
    args['log_file'] = log_file
    with open(args['log_file'], 'a') as f:
        print(f"Params to select: {param_dict}", file=f, flush=True)
    args['mode'] = 'train'
    args['repeat'] = 3
    # utils.print_config(args['log_file'], args)
    performance = {'f1': 0, 'idx': 0, 'std': 0}
    for i, param_set in enumerate(list(ParameterGrid(param_dict))):
        torch.cuda.empty_cache()
        f1_list = []
        args.update(param_set)
        encoder = MultiPPI_Encoder(args)
        with open(args['log_file'], 'a') as f:
            print("Combination {}:".format(i), file=f, flush=True)
            for key, val in param_set.items():
                print(key, ':', val, file=f)
        for _ in range(args['repeat']):
            f1, _ = encoder.train_single_fold(4, print_args=False)
            f1_list.append(f1)
        f1 = np.average(f1_list)
        std = np.std(f1_list)
        with open(args['log_file'], 'a') as f:
            print(f"Avg F1: {f1:.4f}±{std:.4f}", file=f, flush=True)
        if f1 > performance['f1']:
            performance.update(f1=f1, idx=i, std=std)
            best_params = param_set

    with open(args['log_file'], 'a') as f:
        print("Best Combination is {}:".format(performance['idx']), file=f)
        for key, val in best_params.items():
            print(key, ':', val, file=f)
        print(f"Valid F1:{performance['f1']:.4f}±{performance['std']:.4f}", file=f, flush=True)


if __name__ == "__main__":
    args = arg_parse()
    configs = config_load.get()
    configs.update(vars(args))
    configs['gpu'] = f'cuda:{args.gpu}'
    configs['fold'] = args.fold
    configs['data_dir'] = args.data_dir
    configs['overlap'] = args.overlap
    configs['model'] = args.model
    configs['wandb'] = False if args.debug else configs['wandb']
    if args.ppi:
        configs['PPI'] = args.ppi
    cell_line = utils.get_cell_line(configs['data_dir'])
    configs['residual'] = 0.1 if cell_line in ['K562'] else configs['residual']
    
    if args.sh:
        log_file = os.path.join(configs['log_dir'], f"{cell_line}_SH_{configs['gpu'].split(':')[-1]}.txt")
        # param_dict = { 'drop_neg' : [0.3, 0.5, 0.7],
        #            'drop_rate' : [0.3, 0.5, 0.7],
        #            'num_neighbors' : [[40, 10]], # [10, 40], [40, 10], [20,20]
        #            'attn_drop' : [0.1, 0.3, 0.5]}
        # param_dict = {'heads': [4],
        #               'weight_decay': [0.0, 0.05],#0.001, 0.005],
        #               'hidden_channels': [24],}
        param_dict = {'pos_lambda': [0.1, 0.01, 0.001],
                      'neg_lambda': [0.1, 0.01]}
                    #   'pooling': ['lin', 'False']}, #['avg', 'max', 'lin', False]
        select_hyperparams(configs, param_dict, log_file)
        sys.exit()
    
    if args.train:
        configs['mode'] = 'train'
        configs['log_file'] = os.path.join(configs['log_dir'], f"{cell_line}_{configs['mode']}_{args.gpu}")
        configs['log_file'] += '_op.txt' if args.overlap else '.txt'
        encoder = MultiPPI_Encoder(configs)
        if configs['model'] == 'SVM':
            encoder.train_SVM()
        else:
            encoder.train()
        sys.exit()

    if args.pred:
        configs['mode'] = 'pred'
        configs['log_file'] = os.path.join(configs['log_dir'], f"{cell_line}_train")
        encoder = MultiPPI_Encoder(configs)
        model_dirs = os.listdir(os.path.join(configs['out_dir'], encoder.log_name,))
        for i, model_dir in enumerate(model_dirs):
            model_dir = os.path.join(configs['out_dir'], encoder.log_name, model_dir)
            known_true, known_score, known_index = encoder.test(
                                                        model_dir=model_dir,
                                                        pred=False,
                                                        output=i,)
            _, unknown_score, unknown_index =    encoder.test(
                                                        model_dir=model_dir,
                                                        pred=True,
                                                        output=i,)
            if i == 0:
                known_mid = np.array([known_index, known_true, known_score]).T
                known_pred = pd.DataFrame(known_mid, columns=['gene_index', 'Label', f'score_{i}'])
                unknown_mid = np.array([unknown_index, unknown_score]).T
                unknown_pred = pd.DataFrame(unknown_mid, columns=['gene_index', f'score_{i}'])
            else:
                known_mid = np.array([known_index, known_score]).T
                known_mid = pd.DataFrame(data=known_mid, columns=['gene_index', f'score_{i}'])
                known_pred = known_pred.merge(known_mid)
                unknown_mid = np.array([unknown_index, unknown_score]).T
                unknown_mid = pd.DataFrame(data=unknown_mid, columns=['gene_index', f'score_{i}'])
                unknown_pred = unknown_pred.merge(unknown_mid)
            
        score_col = [f'score_{i}' for i in range(configs['cv_folds'])]
        known_pred['avg_score'] = known_pred[score_col].mean(axis=1)
        known_pred['pred_label'] = known_pred.apply(
            lambda x: 1 if x['avg_score'] > 0.5 else 0, axis=1)
        unknown_pred['avg_score'] = unknown_pred[score_col].mean(axis=1)
        unknown_pred['pred_label'] = unknown_pred.apply(
            lambda x: 1 if x['avg_score'] > 0.5 else 0, axis=1)
        
        cell_line = utils.get_cell_line(configs["data_dir"])
        gene_list = utils.get_gene_list()
        gene_list['gene_index'] = np.arange(gene_list.shape[0])
        known_pred = gene_list.merge(known_pred)
        known_pred = known_pred.drop('gene_index', axis=1)
        unknown_pred = gene_list.merge(unknown_pred)
        unknown_pred = unknown_pred.drop('gene_index', axis=1)
        known_pred.to_csv(f'data/Results/{cell_line}_known_pred.csv', index=False)
        unknown_pred.to_csv(f'data/Results/{cell_line}_unknown_pred.csv', index=False)
            
    
    if args.ablation:
        cell_line = utils.get_cell_line(configs['data_dir'])
        configs['mode'] = 'train'
        # configs['start_epoch'] = 0
        ablate_dict = {
            'pos_lambda': [0.,],
            'neg_lambda': [0.,],
            'residual': [0.],
            'learnable_factor': [False,],
        }
        configs['log_file'] = os.path.join(configs['log_dir'], f"{cell_line}_ablation_{args.gpu}.txt")
        with open(configs['log_file'], 'a') as f:
                print(f"Ablation: {ablate_dict}", file=f, flush=True)
        for i, param_set in enumerate(list(ParameterGrid(ablate_dict))):
            configs.update(param_set)
            with open(configs['log_file'], 'a') as f:
                print(f"Settings: {param_set}", file=f, flush=True)
            encoder = MultiPPI_Encoder(configs)
            encoder.train()
        sys.exit()
    
    if args.encode:
        cell_line = utils.get_cell_line(configs['data_dir'])
        mode = 'train'
        configs['log_file'] = os.path.join(configs['log_dir'], f"{cell_line}_{mode}.txt")
        configs['mode'] = 'encode'
        encoder = MultiPPI_Encoder(configs)
        out_dir = f'{cell_line}_embedding.csv'
        encoder.encode(out_dir)
        sys.exit()