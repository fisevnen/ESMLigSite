import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchsnooper

class QKVGenerator(nn.Module):
    def __init__(self, n_head, model_d, k_d, v_d):
        super().__init__()

        self.n_head = n_head
        self.k_d = k_d #dimension
        self.v_d = v_d
        
        self.w_q = nn.Linear(model_d, n_head * k_d, bias=False) # dimension of Q is the same as K
        self.w_k = nn.Linear(model_d, n_head * k_d, bias=False)
        self.w_v = nn.Linear(model_d, n_head * v_d, bias=False)
        self.fc = nn.Linear(n_head * v_d, model_d, bias=False)

    def forward(self, q, k, v):
        k_d, v_d, n_head = self.k_d, self.v_d, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) #sz_b:batch_size len:sequence length

        q = self.w_q(q).view(sz_b, len_q, n_head, k_d)
        k = self.w_k(k).view(sz_b, len_k, n_head, k_d)
        v = self.w_v(v).view(sz_b, len_v, n_head, v_d)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3) #to calculate scaled dot-product

        return q, k, v
    
class QKVGenerator_single(nn.Module):
    def __init__(self, model_d, k_d, v_d):
        super().__init__()

        # self.k_d = k_d #dimension
        # self.v_d = v_d
        
        self.w_q = nn.Linear(model_d, k_d, bias=False) # dimension of Q is the same as K
        self.w_k = nn.Linear(model_d, k_d, bias=False)
        self.w_v = nn.Linear(model_d, v_d, bias=False)
        # self.fc = nn.Linear(n_head * v_d, model_d, bias=False)

    def forward(self, q, k, v):
        q1 = self.w_q(q)
        k1 = self.w_k(k)
        v1 = self.w_v(v)

        return q1, k1, v1

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hide, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_in, eps=1e-6)
        self.fc1 = nn.Linear(d_in, d_hide)
        self.fc2 = nn.Linear(d_hide, d_hide)
        self.ln2 = nn.LayerNorm(d_hide, eps=1e-6)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x1 = self.ln1(x)
        residual = x1
        x2 = self.fc2(F.relu(self.fc1(x1)))
        x3 = self.dropout(x2)
        x5 = self.ln2(x3)

        return x5

def DualContactBulid(tensor1, tensor2):
    zeros_for_tensor1 = torch.zeros(tensor1.size(0), tensor1.size(-1), tensor2.size(-1))
    zeros_for_tensor2 = torch.zeros(tensor1.size(0), tensor2.size(-1), tensor1.size(-1))
    top_row = torch.cat((tensor1, zeros_for_tensor1), dim=2)
    bottom_row = torch.cat((zeros_for_tensor2, tensor2), dim=2)
    concatenated_tensor = torch.cat((top_row, bottom_row), dim=-2)
    return concatenated_tensor

def GlobalNormalize(tensor_in):
    assert len(tensor_in.shape) == 3, 'Length of input tensor is not 3'
    matrix_list = []
    for num in range(tensor_in.shape[0]):
        matrix = tensor_in[num]
        mean = matrix.mean()
        std = matrix.std()
        normalized_matrix = F.normalize(matrix, mean, std)
        matrix_list.append(normalized_matrix)
    tensor_out = torch.stack(matrix_list, dim=0)
    return tensor_out


class focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes  
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = self.alpha[0] + alpha
            self.alpha[1:] = self.alpha[1:] + (1-alpha) 
        self.gamma = gamma

    def forward(self, preds, labels):     
        preds = preds.contiguous().view(-1,preds.size(-1))        
        # print(preds.shape)
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=-1) 
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))    
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class DataFrameCustomDataset(Dataset):
    def __init__(self, PDBID_list, csv_file, label_npy):
        self.PDBID_index = PDBID_list
        self.data = pd.read_csv(csv_file)
        self.label = np.load(label_npy, allow_pickle=True).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        PDBID = self.PDBID_index[idx]
        window_data = self.data[self.data['name'] == PDBID]
        # print(PDBID)
        # print(window_data)
        label1 = self.label[PDBID]
        return (window_data, label1)
    
class DataFrameCustomDataset_affinity_extracted(Dataset): # writing
    def __init__(self, index_list, csv_file, extracted_path):
        self.index_list = index_list
        self.data = pd.read_csv(csv_file, index_col='new_index')
        self.esm_path = extracted_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index = int(self.index_list[idx])
        window_data = self.data[self.data['new_new_index'] == index]
        # print(window_data)
        uID = window_data['UniProt (SwissProt) Primary ID of Target Chain'].iloc[0]
        rep_ = np.load(f'{self.esm_path}/{uID}_rep.npy')
        attention_ = np.load(f'{self.esm_path}/{uID}_attention.npy')
        contact_ = np.load(f'{self.esm_path}/{uID}_contact.npy')
        label1 = self.data.loc[index, '-log affnity']
        return (window_data, 
                rep_, 
                attention_, 
                contact_, 
                label1)

class DataFrameCustomDataset_affinity(Dataset):
    def __init__(self, index_list, csv_file):
        self.index_list = index_list
        self.data = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index = int(self.index_list[idx])
        window_data = self.data[self.data['new_new_index'] == index]
        label1 = self.data.loc[index, '-log affnity']
        return (window_data, label1)
    
class DataFrameCustomDataset_MTL(Dataset):
    def __init__(self, csv_file, contact_map_npy):
        
        self.data = pd.read_csv(csv_file, index_col='num_index')
        self.contact_map_label_data = np.load(contact_map_npy, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['index'].tolist())

    def __getitem__(self, idx):
        # index = int(self.index_list[idx])
        window_data = self.data[self.data['new_num_index'] == idx]
        affinity_label = self.data.loc[idx, 'normalized affinity']
        site_label = self.data.loc[idx, 'site_label'] 
        contact_map_label = self.contact_map_label_data[self.data.loc[idx, 'index'] + '_distance']
        return (window_data, affinity_label, site_label, contact_map_label)

class DataFrameCustomDataset_MTL_extracted(Dataset):
    def __init__(self, index_list, csv_file, contact_map_npy, embedding_dict):
        self.embedding_dict = embedding_dict
        self.index_list = index_list 
        self.data = pd.read_csv(csv_file, index_col='num_index')
        self.contact_map_label_data = np.load(contact_map_npy, allow_pickle=True).item()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = int(self.index_list[idx])
        window_data = self.data[self.data['new_num_index'] == index]
        affinity_label = self.data.loc[index, 'normalized affinity']
        site_label = self.data.loc[index, 'site_label'] 
        contact_map_label = self.contact_map_label_data[self.data.loc[index, 'index'] + '_distance']
        embedding = self.embedding_dict[self.data.loc[index, 'index']].detach()
        return (window_data, affinity_label, site_label, contact_map_label, embedding)

def double_padding(input_tensor, pro_len, pro_padding_len, lig_padding_len):
    pro_inter = pad(input_tensor[:pro_len, :pro_len], (0, pro_padding_len, 0, pro_padding_len))
    lig_inter = pad(input_tensor[pro_len:, pro_len:], (0, lig_padding_len, 0, lig_padding_len))
    right_upper = pad(input_tensor[:pro_len, pro_len:], (0, lig_padding_len, 0, pro_padding_len))
    left_lower = pad(input_tensor[pro_len:, :pro_len], (0, pro_padding_len, 0, lig_padding_len))
    # print(pro_inter.shape, lig_inter.shape, right_upper.shape, left_lower.shape)
    upper = torch.cat((pro_inter, right_upper), dim=1)
    lower = torch.cat((left_lower, lig_inter), dim=1)
    out_tensor = torch.cat((upper, lower), dim=0)
    return out_tensor

def DataFrameCollateFn(batch_data):
    data, label = list(zip(*batch_data))
    data = pd.concat(data,axis=0,ignore_index=True)
    # print(data)
    labels = []
    max_pro_len = max(data['protein_len'])
    max_lig_len = max(data['ligand_HA'])
    for index in range(len(label)):
        pro_len = data.iloc[index]['protein_len']
        pro_padding_len = max_pro_len - data.iloc[index]['protein_len']
        lig_padding_len = max_lig_len - data.iloc[index]['ligand_HA']
        labels.append(double_padding(label[index], pro_len, pro_padding_len, lig_padding_len))
        
    padded_tensors = [torch.unsqueeze(i, dim=0) for i in labels]
    combined_tensor = torch.cat(padded_tensors, dim=0)
    return (data, combined_tensor.float())

def DataFrameCollateFn_affnity(batch_data):
    data, label = list(zip(*batch_data))
    data = pd.concat(data,axis=0,ignore_index=True)
    labels_tensor = torch.Tensor(label).float()
    return (data, labels_tensor)

def pad_strings(strings):
    max_len = max(len(s) for s in strings)
    padded_strings = []
    for s in strings:
        padded_strings.append([s + "0" * (max_len - len(s))])
    result = [[int(x) for x in sublist[0]] for sublist in padded_strings]        
    tensor = torch.LongTensor(result)
    return tensor

def DataFrameCollateFn_MTL(batch_data):
    window_data, affinity_label, site_label, contact_map_label = list(zip(*batch_data))
    data = pd.concat(window_data,axis=0,ignore_index=True)
    affinity_labels_tensor = torch.Tensor(affinity_label).float()
    site_label_len_list = [len(i) for i in site_label]
    max_site_label_len = max(site_label_len_list)
    site_labels_tensor = pad_strings(site_label)
    contact_map_lens = [i.shape[0] for i in contact_map_label]
    max_contact_map_len = max(contact_map_lens)
    new_contact_map_list = []
    len_list = [] #[[ori_pro_len, max_pro_len, ori_lig_HA, ori_complex_len],[ori_pro_len, max_pro_len, ori_lig_HA, ori_complex_len]..]
    i = 0
    for contact_map in contact_map_label:
        tmp = []
        # print(max_contact_map_len, contact_map.shape)
        padding_params = (0,max_contact_map_len-contact_map.shape[0],0,max_contact_map_len-contact_map.shape[0])
        out = F.pad(torch.Tensor(contact_map), padding_params, "constant", 0)
        new_contact_map_list.append(out)
        tmp.append(contact_map.shape[0])
        len_list.append([site_label_len_list[i], max_site_label_len, data.iloc[i,5], contact_map.shape[0]])
        i = i + 1
    out_contact_map = torch.stack(new_contact_map_list, dim=0)

    return (data, affinity_labels_tensor, site_labels_tensor, out_contact_map, len_list)

def DataFrameCollateFn_MTL_extracted(batch_data):
    window_data, affinity_label, site_label, contact_map_label, esm_embedding = list(zip(*batch_data))
    data = pd.concat(window_data,axis=0,ignore_index=True)
    affinity_labels_tensor = torch.Tensor(affinity_label).float()
    site_label_len_list = [len(i) for i in site_label]
    max_site_label_len = max(site_label_len_list)
    site_labels_tensor = pad_strings(site_label)
    contact_map_lens = [i.shape[0] for i in contact_map_label]
    max_contact_map_len = max(contact_map_lens)
    new_contact_map_list = []
    len_list = [] #[[ori_pro_len, max_pro_len, ori_lig_HA, ori_complex_len],[ori_pro_len, max_pro_len, ori_lig_HA, ori_complex_len]..]
    i = 0
    for contact_map in contact_map_label:
        tmp = []
        # print(max_contact_map_len, contact_map.shape)
        padding_params = (0,max_contact_map_len-contact_map.shape[0],0,max_contact_map_len-contact_map.shape[0])
        out = F.pad(torch.Tensor(contact_map), padding_params, "constant", 0)
        new_contact_map_list.append(out)
        tmp.append(contact_map.shape[0])
        len_list.append([site_label_len_list[i], max_site_label_len, data.iloc[i,5], contact_map.shape[0]])
        i = i + 1
    out_contact_map = torch.stack(new_contact_map_list, dim=0)    
    
    #esm embedding padding
    max_embedding_len = max([i.shape[1] for i in esm_embedding])
    # print(f'max_embedding_len: {max_embedding_len}')
    new_embeddings = []
    for embedding in esm_embedding:
        padding_params = (0,0,0,max_embedding_len-embedding.shape[1], 0, 0)
        out = F.pad(embedding, padding_params, "constant", 0)
        new_embeddings.append(out)
    out_embedding = torch.stack(new_embeddings, dim=0)
    # print(out_embedding.shape)
    
    return (data, affinity_labels_tensor, site_labels_tensor, out_contact_map, len_list, torch.squeeze(out_embedding, dim=1))


class WeightedMSELoss(nn.Module):
    def __init__(self, threshold, weight: float = 2.0):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction='none')
        self.threshold = threshold

    def forward(self, predictions, targets):
        # Compute the element-wise MSE loss
        mse = self.mse_loss(predictions, targets)
        
        # Create a weight tensor where elements with targets < 0.5 are scaled by self.weight
        weights = torch.where(targets < self.threshold, self.weight, 1.0)
        
        # Apply the weights to the MSE loss
        weighted_mse = mse * weights
        
        # Return the mean of the weighted MSE loss
        return weighted_mse.mean()

class DistanceMapLoss(nn.Module):
    def __init__(self, alpha=3, loss_fn1=WeightedMSELoss(threshold=0.3, weight=1.5), loss_fn2=WeightedMSELoss(threshold=0.45, weight=1.5)):
        super(DistanceMapLoss, self).__init__()    
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.KLloss = torch.nn.KLDivLoss(reduction='mean')
        self.alpha = alpha

    def forward(self, ori_pred, true, len_list, model_parameters):
        total_loss = 0.0
        for i_, _ in enumerate(len_list):
            sub_lens = len_list[i_] #
            pred = ori_pred.clone()
            
            # prediction
            new_ori_pred_protein = pred[i_,1:sub_lens[0]+1,1:sub_lens[0]+1]
            new_ori_pred_ligand = pred[i_,sub_lens[1]:sub_lens[1]+sub_lens[2],sub_lens[1]:sub_lens[1]+sub_lens[2]]
            new_ori_pred_protein_ligand = pred[i_,1:sub_lens[0]+1,sub_lens[1]:sub_lens[1]+sub_lens[2]]
            
            # true labels
            new_true_protein = true[i_, :sub_lens[1], :sub_lens[1]]
            new_true_ligand = true[i_, sub_lens[1]:, sub_lens[1]:]
            new_true_protein_ligand = true[i_, :sub_lens[1], sub_lens[1]:]
            
            new_ori_pred_total = torch.cat([
                torch.cat([new_ori_pred_protein, new_ori_pred_protein_ligand], dim=1),
                torch.cat([torch.transpose(new_ori_pred_protein_ligand, 1, 0), new_ori_pred_ligand], dim=1)
            ], dim=0)
            new_true_total = true[i_, :sub_lens[-1], :sub_lens[-1]]
            
            loss = self.loss_fn1(new_ori_pred_total, new_true_total)
            
            pred_varience = torch.var(new_ori_pred_total)
            true_varience = torch.var(new_true_total)
            loss2 = torch.abs(true_varience - pred_varience) # varience loss
            total_loss = total_loss + loss
        return total_loss
        
class CustomThreshold_new(nn.Module):
    def __init__(self, initial_threshold, initial_value):
        super(CustomThreshold_new, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
        self.value = initial_value

    def forward(self, x):
        return nn.functional.threshold(x, self.threshold.item(), self.value)
    
class CustomThreshold(nn.Module):
    def __init__(self, initial_threshold, initial_value):
        super(CustomThreshold, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
        self.value = nn.Parameter(torch.tensor(initial_value))

    def forward(self, x):
        return nn.functional.threshold(x, self.threshold.item(), self.value.item())

def DataFrameCollateFn_affnity_extracted(batch_data):
    data, rep_, attention_, contact_, label = list(zip(*batch_data))
    new_data = pd.concat(data)
    len_list = new_data['protein_seq'].map(lambda x: len(x)).tolist()
    max_pro_len = max(len_list)
    padded_residue_embedding = []
    padded_contact_matrix = []
    padded_affinity_matrix = []
    for i in range(len(len_list)):
        padding_len = max_pro_len - len_list[i]
        padding_residue = (0, 0, 0, padding_len)
        padding_contact = (1, padding_len + 1, 1, padding_len + 1)
        padding_affinity = (0, padding_len, 0, padding_len)
        padded_residue_embedding_i = torch.nn.functional.pad(
            torch.from_numpy(rep_[i]), padding_residue, "constant", 0
        )
        padded_contact_matrix_i = torch.nn.functional.pad(
            torch.from_numpy(np.squeeze(contact_[i])), padding_contact, "constant", 0
        )
        padded_affinity_matrix_i = torch.nn.functional.pad(
            torch.from_numpy(attention_[i]), padding_affinity, "constant", 0
        )
        padded_residue_embedding.append(padded_residue_embedding_i)
        padded_contact_matrix.append(padded_contact_matrix_i)
        padded_affinity_matrix.append(padded_affinity_matrix_i)
        
    out_residue_embedding = torch.stack(padded_residue_embedding)
    out_contact_matrix = torch.stack(padded_contact_matrix).float()
    out_affinity_matrix = torch.stack(padded_affinity_matrix)
    
    data = pd.concat(data,axis=0,ignore_index=True)
    labels_tensor = torch.Tensor(label).float()
    return (data, out_residue_embedding, out_contact_matrix, out_affinity_matrix, labels_tensor)

def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets)**2))

def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def r_squared(predictions, targets):
    ss_residual = torch.sum((targets - predictions)**2)
    ss_total = torch.sum((targets - torch.mean(targets))**2)
    return 1 - (ss_residual / ss_total)

def pearson_corr(predictions, targets):
    covar = torch.mean((predictions - torch.mean(predictions)) * (targets - torch.mean(targets)))
    std_dev_predictions = torch.std(predictions)
    std_dev_targets = torch.std(targets)
    return covar / (std_dev_predictions * std_dev_targets)

def ligand_embedding_cut(original_embedding, len_list):
    result_tensors = []

    max_size = max(len_list)
    start_idx = 0
    for size in len_list:
        end_idx = int(start_idx + size)
        sub_tensor = original_embedding[start_idx:end_idx, :]

        if sub_tensor.size(0) < max_size:
            padding_rows = max_size - sub_tensor.size(0)
            padding_zeros = torch.zeros((int(padding_rows), int(original_embedding.size(1))))
            sub_tensor = torch.cat([sub_tensor, padding_zeros])

        result_tensors.append(torch.unsqueeze(sub_tensor, dim=0))
        start_idx = end_idx

    concatenated_tensor = torch.cat(result_tensors, dim=0)
    return(concatenated_tensor)

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_layers, block_type=BasicBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, 64, num_layers[0])
        self.layer2 = self._make_layer(block_type, 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, num_layers[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, num_layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.get_channel_size(), num_classes)

    def _make_layer(self, block_type, channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != channels * block_type.get_channel_size():
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, channels * block_type.get_channel_size(), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block_type.get_channel_size()),
            )

        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block_type(self.in_planes, channels, stride, downsample))
            else:
                layers.append(block_type(channels, channels))
            self.in_planes = channels * block_type.get_channel_size()

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x

class ResNetV2_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetV2_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out

class ReciprocalLogCoshLoss(nn.Module):
    def __init__(self, reg_lambda=1e-3, reciprocal=False, dim=None):
        super(ReciprocalLogCoshLoss, self).__init__()
        self.dim = dim
        self.reciprocal = reciprocal
        self.reg_lambda = reg_lambda

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        cosh_diff = torch.cosh(diff)
        log_cosh_diff = torch.log(cosh_diff)
        if self.reciprocal:
            loss = 1.0 / log_cosh_diff
        else: 
            loss = log_cosh_diff
        
        if self.dim is not None:
            return torch.mean(loss, dim=self.dim)
        else:
            return torch.mean(loss)

def embedding_mat(input_tensor):  #[B, r, f] -> [B, r, r, f]
    input_tensor_expanded1 = input_tensor.unsqueeze(2)  # [j, i, 1, 128]
    input_tensor_expanded2 = input_tensor.unsqueeze(1)  # [j, 1, i, 128]
    output_tensor = input_tensor_expanded1 + input_tensor_expanded2  # [j, i, i, 128]
    return output_tensor

