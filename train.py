import torch
import numpy as np
import time
import sys
import torch.optim as optim
from sublayers import *
from inter_net import *
from geminimol.model.GeminiMol import *
from torch.utils.data import DataLoader
from metabalance import MetaBalance
from GradSurgery import GradSurgery
from tensorboardX import SummaryWriter
from torch.nn import init
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

def print_memory_usage(data_df):
    print('=' * 50)
    print('Memory usage:')
    print('=' * 50)
    print('Allocated:', torch.cuda.memory_allocated() / 1024 ** 2, 'MB')
    print('Reserved:', torch.cuda.memory_reserved() / 1024 ** 2, 'MB')
    print('Cached:', torch.cuda.memory_cached() / 1024 ** 2, 'MB')
    if (torch.cuda.memory_allocated() / 1024 ** 2) > 18000:
        print(f'data_df:{data_df}')
    print('Max used:', torch.cuda.max_memory_allocated() / 1024 ** 2, 'MB')
    print('=' * 50)

class ConcatNet(torch.nn.Module):
    def __init__(
        self, 
        args,
        ):
        super((ConcatNet), self).__init__()
        self.Integrate_Net = TestNet(
            Inner_dim = args['multi_head_attention_dim'],
            num_heads = args['multi_head_attention_head'],
        )
        for m in self.Integrate_Net.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                # init.xavier_normal(m.bias.data)
    
    def forward(self, data_df, protein_embedding):
        out_distance_map, out_site = self.Integrate_Net(data_df, protein_embedding)
        return out_distance_map, out_site

    def get_last_shared_layer(self):
        # return self.Integrate_Net.MHA_Cross_layer
        return self.Integrate_Net.Transition
    
class MTL_train(torch.nn.Module):
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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args_json_file = sys.argv[1] # json file which records the args
    args_num = args_json_file.split('/')[-1].split('.')[0].replace('MTL_', '')
    args = read_json_args(args_json_file) # all args
    
    all_embedding = np.load('../Data/PDBbind2020_ESM650M_embedding.npy', allow_pickle=True).item()
    
    train_id_list = fread('../Data/PDBbindv2020/PDBbind_coach_clstr0.4_train_list_500_515')
    test_id_list = fread('../Data/PDBbindv2020/PDBbind_coach_clstr0.4_test_list_500_515')
    
    train_data = DataFrameCustomDataset_MTL_extracted(train_id_list, '../Data/PDBbindv2020/new_full_PDBbind_data_short_noNAN.csv', '../Data/PDBbindv2020/new_PDBbind_distance_map_full.npy', all_embedding)
    test_data = DataFrameCustomDataset_MTL_extracted(test_id_list, '../Data/PDBbindv2020/new_full_PDBbind_data_short_noNAN.csv', '../Data/PDBbindv2020/new_PDBbind_distance_map_full.npy', all_embedding)
    #test_data = DataFrameCustomDataset_affinity(test_index_list, '../Data/new_Test.csv')
    
    train_dataloader = DataLoader(train_data, batch_size = 1, shuffle = True,  collate_fn = DataFrameCollateFn_MTL_extracted)
    test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = False,  collate_fn = DataFrameCollateFn_MTL_extracted)
        
    ConcatenateNet = MTL_train(ConcatNet(args=args), args=args)
    ConcatenateNet.to(device)
    
    writer = SummaryWriter(f'../tb/tb_{args_num}_weights_more_lr/')

    for epoch in range(args['lr_decay_epoch'] * 3):
        # train
        loss_list = train_loop(train_dataloader, device, ConcatenateNet, epoch, args, writer)
        total_loss1 = sum([i[0] for i in loss_list])
        total_loss2 = sum([i[1] for i in loss_list])
        print(f'total contact map loss during trainning process of epoch {epoch}: {total_loss1}')
        print(f'total binding site loss during trainning process of epoch {epoch}: {total_loss2}')
        torch.save(ConcatenateNet.state_dict(), f'MTL_{epoch}_GradNorm_{args_num}.pt')
        torch.cuda.empty_cache()
        # test
        # if epoch > 9:
        test_loss_list = test_loop(test_dataloader, device, ConcatenateNet)
        total_test_loss1 = sum([i[0] for i in test_loss_list])
        total_test_loss2 = sum([i[1] for i in test_loss_list])
        print(f'total contact map loss during test process of epoch {epoch}: {total_test_loss1}')
        print(f'total binding site loss during test process of epoch {epoch}: {total_test_loss2}')
    
