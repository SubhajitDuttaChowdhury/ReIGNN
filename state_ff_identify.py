#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import time
from datetime import datetime
import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataLoader_1_1 import PygNodePropPredDataset
import pandas as pd
import os
import argparse
import time
import copy
import random

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
    
def train(dataset, task, writer, dev, test_ind, bound, netlist_name, keyword):
    batch_raw = pd.read_csv('dataset/stateFlipflopIdentify/raw/node_graph_number.csv', header = None).values
    batch = np.array([i for item in batch_raw for i in item])
    batch = torch.from_numpy(batch).to(torch.long)
    
    max_graph_index = max(batch)
    graph_index = list(range(max_graph_index + 1))
    print("Complete graph index: ", graph_index)
    
    test_netlist_slice = test_ind
    if 0 in test_netlist_slice:
        test_netlist_slice.remove(0)
    print("test netlist slice: ",test_netlist_slice)
    for i in test_netlist_slice:
        print(netlist_name[i])

    train_netlist_slice_int = list(set(graph_index)-set(test_netlist_slice))    
    train_netlist_slice = random.sample(train_netlist_slice_int, int(0.8*max_graph_index))
    if 0 in train_netlist_slice:
            train_netlist_slice.remove(0)
    print("train graph index: ", train_netlist_slice)
    for i in train_netlist_slice:
        print(netlist_name[i])
    validation_netlist_slice = list(set(train_netlist_slice_int)-set(train_netlist_slice))
    if 0 in validation_netlist_slice:
            validation_netlist_slice.remove(0)
    print("validation graph index: ", validation_netlist_slice)
    for i in validation_netlist_slice:
        print(netlist_name[i])
    


    loader = DataLoader(dataset[train_netlist_slice], batch_size=4, shuffle=True)
    loader_validate = DataLoader(dataset[validation_netlist_slice], batch_size=32, shuffle=False)
    loader_test = DataLoader(dataset[test_netlist_slice], batch_size = 32, shuffle = False)


    # build model
    model = GNNStack(max(dataset.num_node_features, 1), 100, dataset.num_classes, task=task).to(dev)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_fitting_model = model
    best_acc = 0
    best_fn = 10000
    
    loss_list = []
    accuracy_list = []

    # train
    for epoch in range(bound):
        total_loss = 0
        model.train()
        correct = 0
        total = 0
        positive = []
        negative = []
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for batch in loader:
            batch=batch.to(dev)
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            pred_dim1 = pred.argmax(dim=1)

            for i in range(len(label)):
                if label[i] == 1:
                    positive.append(pred_dim1[i])
                elif label[i] == 0:
                    negative.append(pred_dim1[i])
            for i in range(len(positive)):
                if positive[i] == 1:
                    true_positive += 1
                else:
                    false_negative += 1
            for i in range(len(negative)):
                if negative[i] == 0:
                    true_negative += 1
                else:
                    false_positive += 1

            weight = torch.tensor([0.006, 0.994]).to(dev)
            loss = model.loss(pred, label, weight)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            correct += pred_dim1.eq(label).sum().item()
        for data in loader.dataset:
            total += torch.sum(data.train_mask).item()
        train_acc = correct / total
        print("##########")
        print ("train accuracy", train_acc)
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)


        if epoch % 1 == 0:
            test_acc, false_negative_test, val_acc, false_negative_val = test(loader_test, loader_validate, model, dev)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)
            print("Epoch {}. Loss: {:.4f}. Validation accuracy: {:.4f}".format(epoch, total_loss, val_acc))

            if (epoch > 10):
                if (false_negative_val < best_fn):
                    best_fitting_model = copy.deepcopy(model)
                    best_acc = val_acc
                    best_fn = false_negative_val
                    print("Best Epoch till now: ", epoch)
                elif ((false_negative_val <= best_fn) and (val_acc > best_acc)):
                        best_fitting_model = copy.deepcopy(model)
                        best_acc = val_acc
                        best_fn = false_negative_val
                        print("Best Epoch till now: ", epoch)

        loss_list.append(total_loss)
        accuracy_list.append(train_acc)
    
    
    os.makedirs("./result/", exist_ok=True)
    modelFileName = "result/model_"+keyword+".txt"
    modelFile = open(modelFileName, "w")
    modelFile.write("best acc: ")
    modelFile.write(str(best_acc))
    modelFile.write("\n")
    modelFile.write("best fn: ")
    modelFile.write(str(best_fn))
    modelFile.write("\n")
    modelFile.write("loss list: \n")
    modelFile.write(str(loss_list))
    modelFile.write("\n")
    modelFile.write("accuracy list: \n")
    modelFile.write(str(accuracy_list))
    modelFile.write("\n")
    print("best acc: ", best_acc)
    torch.save(best_fitting_model.state_dict(), "./best_model_"+keyword+".pt")
    return best_fitting_model, test_netlist_slice


def test(loader_test,loader_validate, model, dev, is_validation=False):
    model.eval()
    start = time.time()
    correct = 0
    for data in loader_test:
        data = data.to(dev)
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            temp_cnt = 0
            for i in pred:
                if i == 1:
                    temp_cnt += 1
            
        correct += pred.eq(label).sum().item()



    positive = []
    negative = []
    for i in range(len(label)):
        if label[i] == 1:
            positive.append(pred[i])
        elif label[i] == 0:
            negative.append(pred[i])
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(len(positive)):
        if positive[i] == 1:
            true_positive += 1
        else:
            false_negative += 1
    for i in range(len(negative)):
        if negative[i] == 0:
            true_negative += 1
        else:
            false_positive += 1
    print("true_positive test: ", true_positive)
    print("false_positive test: ", false_positive)
    print("true_negative test: ", true_negative)
    print("false_negative test: ", false_negative)
    
    if model.task == 'graph':
        total = len(loader_test.dataset) 
    else:
        total = 0
    #total = len(loader.dataset.data.test_mask) 
    for data in loader_test.dataset:
        total += torch.sum(data.test_mask).item()
    print("total:", total)
    stop_time = time.time()
    test_time = stop_time - start
    print("Test time is %.2f", test_time)
    
    
    correct_val = 0
    for data in loader_validate:
        data = data.to(dev)
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask
            pred = pred[mask]
            label = data.y[mask]
            temp_cnt = 0
            for i in pred:
                if i == 1:
                    temp_cnt += 1
            
        correct_val += pred.eq(label).sum().item()


    positive_val = []
    negative_val = []
    for i in range(len(label)):
        if label[i] == 1:
            positive_val.append(pred[i])
        elif label[i] == 0:
            negative_val.append(pred[i])
    true_positive_val = 0
    false_positive_val = 0
    true_negative_val = 0
    false_negative_val = 0
    for i in range(len(positive_val)):
        if positive_val[i] == 1:
            true_positive_val += 1
        else:
            false_negative_val += 1
    for i in range(len(negative_val)):
        if negative_val[i] == 0:
            true_negative_val += 1
        else:
            false_positive_val += 1
    print("true_positive val: ", true_positive_val)
    print("false_positive val: ", false_positive_val)
    print("true_negative val: ", true_negative_val)
    print("false_negative val: ", false_negative_val)
    
    if model.task == 'graph':
        total_val = len(loader_validate.dataset) 
    else:
        total_val = 0
    for data in loader_validate.dataset:
        total_val += torch.sum(data.val_mask).item()
    print("total_val:", total_val)
    
    
    
    
    
    return correct / total, false_negative, correct_val / total_val, false_negative_val


if __name__ == '__main__':
    ## Argument parser
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--epoch', type=int,default=20, help="number of training epochs")
    parser.add_argument('--keyword', type=str, default="aes", help="keyword for the name of the netlist in the testing dataset")
    parser.add_argument('--cuda', type=str, default="cuda:0", help="the cuda device used for training and inference")
    args = parser.parse_args()

    bound = args.epoch

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


    ## Loading training model
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dev = torch.device(str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(dev)
    print('CUDA availability: ', torch.cuda.is_available())
    dataset = PygNodePropPredDataset(name="stateFlipflopIdentify")
    print(dataset)
    task = 'node'
    starttime = time.time()
    model, test_netlist_slice = train(dataset, task, writer, dev, test_ind, bound, netlist_name, args.keyword)
    stoptime = time.time()
    print("train time is: ", stoptime-starttime)
    stat = open("./result/training_stat.txt", 'w')
    stat.write("Train time is: \n")
    train_time = stoptime-starttime
    stat.write(str(train_time))
    stat.write("\n")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total parameters: ",pytorch_total_params)
    stat.write("Total Model Parameters: \n")
    stat.write(str(pytorch_total_params))
    stat.write("\n")
    stat.close()


    color_list = ["red", "orange", (1.0, 1.0, 1.0, 0.001)]
 
    ## Evaluate on each testing netlist
    for netid in test_netlist_slice:  ### test_ind -- the indices of the netlist used for testing.
        loader = DataLoader(dataset[0, netid], batch_size=64, shuffle=False)  #### this 0 is always given since the first index has no graph.
        print(netlist_name[netid])
        embs = []
        colors = []
        for batch in loader:
            batch = batch.to(dev)
            emb, pred = model(batch)
            embs.append(emb.to('cpu'))
            colors += [color_list[y] for y in batch.y]
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

            idFileName = "result/" + str(netlist_name[netid]) + "_idlist.txt"
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

            nameFileName = "result/" + str(netlist_name[netid]) + "_namelist.txt"
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
    embs = torch.cat(embs, dim=0)

