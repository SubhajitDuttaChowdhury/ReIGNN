from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero
from torch_geometric.data import Data
import sys


class PygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform=None, meta_dict = None):

        self.name = name ## original name, e.g., ogbn-proteins

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21
        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(PygNodePropPredDataset, self).__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels


    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        file_names = ['edge']
        file_names.append('node-feat')
        return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('data_processed')

    def split(self, data, batch):
        #print("split")
        node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
        #print(node_slice)
        node_slice = torch.cat([torch.tensor([0]), node_slice])
        #print(node_slice)

        train_mask_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
        #print(train_mask_slice)
        train_mask_slice = torch.cat([torch.tensor([0]), train_mask_slice])
        #print(train_mask_slice)
    
        test_mask_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
        #print(test_mask_slice)
        test_mask_slice = torch.cat([torch.tensor([0]), test_mask_slice])
        #print(test_mask_slice)

        valid_mask_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
        #print(test_mask_slice)
        valid_mask_slice = torch.cat([torch.tensor([0]), valid_mask_slice])
        #print(test_mask_slice)

        row, _ = data.edge_index
        edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
        #print(edge_slice)
        edge_slice = torch.cat([torch.tensor([0]), edge_slice])
        #print(edge_slice)
    
        # Edge indices should start at zero for every graph.
        data.edge_index -= node_slice[batch[row]].unsqueeze(0)
        data.__num_nodes__ = torch.bincount(batch).tolist()
    
        slices = {'edge_index': edge_slice}
        if data.x is not None:
            slices['x'] = node_slice
        if data.train_mask is not None:
            slices['train_mask'] = train_mask_slice
        if data.test_mask is not None:
            slices['test_mask'] = test_mask_slice
        if data.val_mask is not None:
            slices['val_mask'] = valid_mask_slice
        if data.edge_attr is not None:
            slices['edge_attr'] = edge_slice
        if data.y is not None:
            if data.y.size(0) == batch.size(0):
                slices['y'] = node_slice
            else:
                slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    
        return data, slices

    def process(self, split_type = None):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

        node_label_raw = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip', header = None).values
        node_label = np.array([i for item in node_label_raw for i in item])
        #print(node_label)
        #print(len(node_label))

        if 'classification' in self.task_type:
            # detect if there is any nan
            if np.isnan(node_label).any():
                data.y = torch.from_numpy(node_label).to(torch.float32)
            else:
                data.y = torch.from_numpy(node_label).to(torch.long)

        else:
            data.y = torch.from_numpy(node_label).to(torch.float32)

        data if self.pre_transform is None else self.pre_transform(data)
        data.train_mask = []
        data.val_mask = []
        #data.test_mask = []
        for i in range(len(node_label)):
            if node_label[i] == 0 or node_label[i] == 1:
                data.train_mask.append(True)
            else:
                data.train_mask.append(False)
            #if node_label[i] == 1:
            #    data.val_mask.append(True)
            #else:
            #    data.val_mask.append(False)
            #if i in test_idx:
            #    data.test_mask.append(True)
            #else:
            #    data.test_mask.append(False)
        data.train_mask = np.array(data.train_mask)
       # data.val_mask = np.array(data.val_mask)
        #data.test_mask = np.array(data.test_mask)
        data.train_mask = torch.from_numpy(data.train_mask).to(torch.bool)
        data.test_mask = data.train_mask
        data.val_mask = data.train_mask
       # data.val_mask = torch.from_numpy(data.val_mask).to(torch.bool)
        #data.test_mask = torch.from_numpy(data.test_mask).to(torch.bool)

        #print(data.train_mask)
        #print(data.val_mask)
        #print(data.test_mask)
        batch_raw = pd.read_csv(osp.join(self.raw_dir, 'node_graph_number.csv'), header = None).values
        batch = np.array([i for item in batch_raw for i in item])
        batch = torch.from_numpy(batch).to(torch.long)
        #print(batch)
        data = Data(x=data.x, edge_index=data.edge_index, test_mask=data.test_mask, train_mask=data.train_mask, val_mask=data.val_mask, y=data.y)
        #print(data)
        data, slices = self.split(data, batch)
        #print(slices)
                
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
        #return '{}({})'.format(self.__class__.__name__, len(self))
        

if __name__ == '__main__':
    print("#------ Instruction ------#")
    print("# Usage: python3 dataLoader_1.0.py [DatasetName]")
    print("# Hint 1: The dataset you want to load should be already put in the folder 'dataset'.")
    print("# Hint 2: In the customized dataset, 'raw' folder is needed, which must includes these files: ")
    print("#             edge.csv.gz, node-feat.csv.gz, node-lable.csv.gz")
    print("# Hint 3: The information of the customized dataset should be added to master.csv.")
    if (len(sys.argv) != 2):
        print("#--------- Error ---------#")
        print("# The number of inputs is illegal!")
        print("#-------------------------#")
        exit()
    print("#------------------------#\n")
    pyg_dataset = PygNodePropPredDataset(name = sys.argv[1])
    print(pyg_dataset[0])
    #pyg_dataset.get_idx_split()
    
