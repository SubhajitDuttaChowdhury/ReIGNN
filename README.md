This repo contains the source code for **ReIGNN: State Register Identification Using Graph Neural Network for Reverse Engineering** (accepted to ICCAD 2021).

**Tool Dependency:**

- Python 3.7
- Pytorch (https://pytorch.org/get-started/locally/) (Please install as per the system)
- Pytorch Geometric (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (Please install as per the system)
- Networkx (https://networkx.org/documentation/stable/install.html)
- Pandas (https://pandas.pydata.org/docs/getting_started/install.html)
- Sklearn (https://scikit-learn.org/stable/install.html) 
- numpy (https://numpy.org/install/)

All necessary files and folders should be present in the main folder. The names and function of these files are as follows:

- dataset folder -- This folder contains the dataset which in our case is a collection of 40 designs. The files associated with the graphs are present in the raw subfolder in the dataset folder.

- There are in total 8 different designs. Each design is synthesized with 5 different operating frequencies to create different versions of it.

- splitted_dataset -- This folder contains information about each design (its nodes names and the SCCs present in it).

- dataLoader_1_1.py -- This script is used for processing the raw dataset.

- master.csv -- This file contains the necessary information used for training the GNN model.

- design_name_in_order -- This file contains the names of the designs present in the dataset following the same order.

- state_ff_identify.py -- This script is used for training the GNN. In this work, we do k-fold cross validation, hence we train a GNN to perform inference on a particular design. For example, if we want to perform inference for aes and its variants then the test set will be the 5 different versions of aes, and the rest of the designs will be divided into train and validation set. 

- test_model.py -- This script is used to load a trained model and perform inference.

- result -- Once training/ inference is done, the output are stored in this folder. For each design, 2 files are created -- design_name_idlist.txt -- which contains the true positive, false positive, true negative, and false negative nodes before structural analysis, design_name_namelist.txt contains the TP, FP, TN, FN nodes after structural analysis.

To perform training, we need to run the following command:

./run_train.sh

If you want to create a trained model for aes2, please specify that in the keyword as aes2, and also specify the number of epochs, and the cuda that should be used. Once the training is done, the best performing model will be saved in this main folder (best_model_keyword) and the performance for each variant of the design will be stored in the result folder. 

The 8 different keywords are: aes2, altor, completogpio, fsm, siphash, cr_div, sha, MEMORY

If you want to perform inference for an already trained model then we need to run the following command:

./run_test.sh

In this file specify the keyword and the cuda.

Each time we train, the weights of the GNN are initialized differently and hence a new trained model is generated. If you want to reproduce the result of our paper, we have to perform inference on a pretrained model.

If you want to perform register classification for some other designs, then it should be first synthesized with Nangate library, and then converted to a graph. Then we need to add it to the exisiting dataset. Please let us know if you would like to add more designs.
