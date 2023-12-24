import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import glob

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch import autograd
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch_geometric.nn as pyg_nn
from tqdm.notebook import tqdm
import torch.optim as optim
from scipy.optimize import minimize
from scipy.special import logit, expit

from scipy.special import softmax
from scipy.special import log_softmax
from sklearn.metrics import log_loss
from dataLoader_1_1 import PygNodePropPredDataset
import argparse

# this part is the definition of the GNN model structure
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        for x in range(2):
            self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp1 = nn.Sequential(
            nn.Linear(hidden_dim, 50))
        self.post_mp2 = nn.Sequential(
            nn.Linear(50, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.SAGEConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        emb = x
        x = self.post_mp1(x)
        x = F.relu(x)
        x = F.dropout(x, p =0.5, training=self.training)
        x = self.post_mp2(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label, weight):
        return F.nll_loss(pred, label, weight=weight)
 


if __name__ == '__main__':
    ## Argument parser
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--keyword', type=str, default="aes", help="keyword for the name of the netlist in the testing dataset")
    parser.add_argument('--model', type=str, default="best_model_gcm_aes", help="specify the model used for testing")
    parser.add_argument('--cuda', type=str, default="cuda:0", help="the cuda device used for training and inference")
    args = parser.parse_args()

    benchmark_name_list = open("design_name_in_order", 'r')
    updated_name_list = [""]
    netlist_ind = 1
    test_ind = []
    for line in benchmark_name_list:
        line = line.replace("\n", "")
        line_upt = line.split(" ")
        for item in line_upt:
            item = item.split("/")[-1]
            item = item.split(".")[0]
            updated_name_list.append(item)
            if args.keyword in item:
                test_ind.append(netlist_ind)
            netlist_ind += 1

    netlist_name = updated_name_list

    dev = torch.device(str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('CUDA availability: ', torch.cuda.is_available())
    dataset = PygNodePropPredDataset(name="stateFlipflopIdentify")
    
    task = 'node'
    model = GNNStack(max(dataset.num_node_features, 1), 100, dataset.num_classes, task=task).to(dev)
    #model.load_state_dict(torch.load(os.getcwd()+"/best_model_gcm_aes.pt"))
    model.load_state_dict(torch.load(os.getcwd()+"/"+args.model))
    print(model)
    
    # test set index
    test_netlist_slice = test_ind
    if 0 in test_netlist_slice:
        test_netlist_slice.remove(0)
    print("test netlist slice: ",test_netlist_slice)
    for i in test_netlist_slice:
        print(netlist_name[i])

    for netid in test_netlist_slice:  ### test_ind -- the indices of the netlist used for testing.
        loader = DataLoader(dataset[0, netid], batch_size=64, shuffle=False)  #### this 0 is always given since the first index has no graph.
        print(netlist_name[netid])
        embs = []
        for batch in loader:
            batch = batch.to(dev)
            emb, pred = model(batch)
            embs.append(emb.to('cpu'))
            label = batch.y.to('cpu')
            pred = pred.argmax(dim=1)
            pred = pred.to('cpu')
            positive_list = []
            negative_list = []
            true_positive_list = []
            false_positive_list = []
            true_negative_list = []
            false_negative_list = []

            true_positive = 0
            false_positive = 0
            true_negative = 0
            false_negative = 0
            
            node_name_mapping = []
            nodeNameFile = open("splited_dataset/node_name/node_names"+netlist_name[netid]+".csv", "r")
            for line in nodeNameFile:
                node_name_mapping.append(str(line).replace('\n', ''))
            nodeNameFile.close()
            for i in range(len(label)):
                if label[i] == 1:
                    positive_list.append(pred[i])
                    if pred[i] == 1:
                        true_positive += 1
                        true_positive_list.append(i)
                    else:
                        false_negative += 1
                        false_negative_list.append(i)
                elif label[i] == 0:
                    negative_list.append(pred[i])
                    if pred[i] == 1:
                        false_positive += 1
                        false_positive_list.append(i)
                    else:
                        true_negative += 1
                        true_negative_list.append(i)

            print("true_positive: ",true_positive)
            print("false_positive: ",false_positive)

            print("true_negative: ",true_negative)
            print("false_negative: ",false_negative)

            os.makedirs("./result/", exist_ok=True)
            idFileName = "result/" + str(netlist_name[netid]) + "_test_idlist.txt"
            idFile=open(idFileName, "w")
            idFile.write("true_positive_list: ")
            idFile.write(str(true_positive_list))
            idFile.write("\n")
            idFile.write("true_positive: ")
            idFile.write(str(true_positive))
            idFile.write("\n")
            idFile.write("false_positive_list: ")
            idFile.write(str(false_positive_list))
            idFile.write("\n")
            idFile.write("false_positive: ")
            idFile.write(str(false_positive))
            idFile.write("\n")
            idFile.write("true_negative_list: ")
            idFile.write(str(true_negative_list))
            idFile.write("\n")
            idFile.write("true_negative: ")
            idFile.write(str(true_negative))
            idFile.write("\n")
            idFile.write("false_negative_list: ")
            idFile.write(str(false_negative_list))
            idFile.write("\n")
            idFile.write("false_negative: ")
            idFile.write(str(false_negative))
            idFile.write("\n")
            idFile.close()

            tplist_ffname = []
            fplist_ffname = []
            tnlist_ffname = []
            fnlist_ffname = []
            for ind in true_positive_list:
                tplist_ffname.append(node_name_mapping[ind])
            for ind in false_positive_list:
                fplist_ffname.append(node_name_mapping[ind])
            for ind in true_negative_list:
                tnlist_ffname.append(node_name_mapping[ind])
            for ind in false_negative_list:
                fnlist_ffname.append(node_name_mapping[ind])

            scc_list = []
            nodeNameFile=open("splited_dataset/scc/"+netlist_name[netid]+"_scc.txt", "r")
            for line in nodeNameFile:
                if line[0] == '{':
                    if line.count(",") > 10:
                        scc_list.append(line.replace('\n', ''))

            toBeDeleted = []
            for ind1 in fplist_ffname:
                ind1_check = False
                for ind2 in scc_list:
                    if str(ind1) in str(ind2):
                        ind1_check = True
                        break
                if ind1_check == False:
                    false_positive -= 1
                    true_negative += 1
                    toBeDeleted.append(ind1)
            for ind in toBeDeleted:
                fplist_ffname.remove(ind)
                tnlist_ffname.append(ind)

            toBeDeleted = []
            for ind1 in tplist_ffname:
                ind1_check = False
                for ind2 in scc_list:
                    if str(ind1) in str(ind2):
                        ind1_check = True
                        break
                if ind1_check == False:
                    true_positive -= 1
                    false_negative += 1
                    toBeDeleted.append(ind1)
            for ind in toBeDeleted:
                tplist_ffname.remove(ind)
                fnlist_ffname.append(ind)

            nameFileName = "result/" + str(netlist_name[netid]) + "_test_namelist.txt"
            nameFile=open(nameFileName, "w")
            nameFile.write("true_positive_list: ")
            nameFile.write(str(tplist_ffname))
            nameFile.write("\n")
            nameFile.write("true_positive: ")
            nameFile.write(str(true_positive))
            nameFile.write("\n")
            nameFile.write("false_positive_list: ")
            nameFile.write(str(fplist_ffname))
            nameFile.write("\n")
            nameFile.write("false_positive: ")
            nameFile.write(str(false_positive))
            nameFile.write("\n")
            nameFile.write("true_negative_list: ")
            nameFile.write(str(tnlist_ffname))
            nameFile.write("\n")
            nameFile.write("true_negative: ")
            nameFile.write(str(true_negative))
            nameFile.write("\n")
            nameFile.write("false_negative_list: ")
            nameFile.write(str(fnlist_ffname))
            nameFile.write("\n")
            nameFile.write("false_negative: ")
            nameFile.write(str(false_negative))
            nameFile.write("\n")
            nameFile.close()

            print("true_positive: ", true_positive)
            print("false_positive: ", false_positive)
            print("true_negative: ", true_negative)
            print("false_negative: ", false_negative)

