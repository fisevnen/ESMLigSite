import torch
import numpy as np
import time
import sys
import torch.optim as optim
from sublayers import *
from ESMLigNet_pytorch_MTL import *
from geminimol.model.GeminiMol import *
from torch.utils.data import DataLoader
from metabalance import MetaBalance
from GradSurgery import GradSurgery
# from torchsummary import summary

def remove_elements(list1, list2,list3,list4):
    set1 = set(list1)
    set2 = set(list2+list3+list4)
    return list(set1 - set2)

def read_json_args(json_file):
    with open(json_file,'r') as load_f:
        load_dict = json.load(load_f)
        load_f.close()
    return load_dict

"""
###############################################################################
#                             CONFIDENTIAL NOTICE                             #
#                                                                             #
# This file contains masked implementations of ESMLigSite model.              #
#                                                                             #
# Certain critical portions of this code have been intentionally obscured.    #
#                                                                             #
# The full, unmasked implementation including:                                #
# - Complete neural network architecture                                      #
# - Advanced feature engineering components                                   #
# - Optimized training procedures                                             #
# - Proprietary integration methods                                           #
#                                                                             #
# will be made publicly available upon publication of our research.           #
#                                                                             #
# For inquiries about licensing or early access, please contact the authors.  #
#                                                                             #
# © 2025 ShanghaiTech University. All rights reserved.                        #
###############################################################################
"""

def pretrained_lr(epoch, lr_decay_epoch, min_lr):
    if epoch <= lr_decay_epoch:
        return 0.0
    elif lr_decay_epoch < epoch <= 2 * lr_decay_epoch:
        return min_lr
    elif 2 * lr_decay_epoch < epoch <= 3 * lr_decay_epoch:
        return 5 * min_lr
    
def decoder_lr(epoch, lr_decay_epoch, min_lr):
    if epoch <= lr_decay_epoch:
        return 10 * min_lr
    elif lr_decay_epoch < epoch <= 2 * lr_decay_epoch:
        return 5 * min_lr
    elif 2 * lr_decay_epoch < epoch <= 3 * lr_decay_epoch:
        return min_lr

    
def test_loop(dataloader, device, model, checkpoint_name):
    model.eval()
    out_list = []
    iter = 0
    with torch.no_grad():
        task_losses = []
        for batch, (input_df,  affinity_labels, site_labels, contact_maps, len_list, esm_embedding) in enumerate(dataloader):
            # print(input_df)
            affinity_labels = affinity_labels.to(device)
            site_labels = site_labels.to(device)
            contact_maps = contact_maps.to(device)
            esm_embedding = esm_embedding.to(device)
            task_loss = model(input_df, esm_embedding, affinity_labels, site_labels, contact_maps, len_list)[1]
            out_list.append([model(input_df, esm_embedding, affinity_labels, site_labels, contact_maps, len_list)[0], task_loss])
            task_losses.append(task_loss.data.cpu().numpy())
            if iter % 100 == 0:
                print('test_iter {}: task_loss={}'.format(
                        iter, task_loss.data.cpu().numpy()), flush=True)

            torch.cuda.empty_cache()
            time.sleep(0.1)
            iter += 1
    torch.save(out_list, f'{checkpoint_name}_test_results.pt')
    return task_losses

# def padding_lig_tensor(original_tensor, HA_list)
def fread(file, removen=True, remove1strow=False):
    with open(file, 'r') as f:
        contents = f.readlines()
        f.close()
    if removen == True:
        contents = [i.replace('\n','') for i in contents]
    if remove1strow == True:
        contents.pop(0)
    return contents

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_json_file = sys.argv[1] # json file which records the args
    args_num = args_json_file.split('/')[-1].split('.')[0].replace('MTL_', '')
    args = read_json_args(args_json_file) # all args
    
    all_embedding = np.load('../Data/PDBbind2020_ESM650M_embedding.npy', allow_pickle=True).item()
    test_id_list = fread('../Data/PDBbindv2020/PDBbind_clstr0.9_test_list')
    
    test_data = DataFrameCustomDataset_MTL_extracted(test_id_list, '../Data/PDBbindv2020/new_full_PDBbind_data_short.csv', '../Data/PDBbindv2020/new_PDBbind_distance_map_full.npy', all_embedding)
    
    test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = False,  collate_fn = DataFrameCollateFn_MTL_extracted)
    
    ConcatenateNet = MTL_train(ConcatNet(args=args), args=args)
    ConcatenateNet.to(device)
    checkpoint_file = sys.argv[2]
    checkpoint = torch.load(checkpoint_file)
    # print(checkpoint)
    ConcatenateNet.load_state_dict(checkpoint)
    checkpoint_name = checkpoint_file.split('/')[-1].replace('.pt', '')
    test_loss_list = test_loop(test_dataloader, device, ConcatenateNet, checkpoint_name) 
    total_test_loss1 = sum([i[0] for i in test_loss_list])
    total_test_loss2 = sum([i[1] for i in test_loss_list])
    total_test_loss3 = sum([i[2] for i in test_loss_list if not np.nan in i])
    # test_loop(test_dataloader, device, ConcatenateNet, epoch)
    