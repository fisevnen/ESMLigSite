import torch
import numpy as np
import time
import sys
from tqdm import tqdm
import esm
import torch.optim as optim
from sublayers import *
from inter_net_inference import *
from geminimol.model.GeminiMol import *
from torch.utils.data import DataLoader


def read_json_args(json_file):
    with open(json_file,'r') as load_f:
        load_dict = json.load(load_f)
        load_f.close()
    return load_dict

class ConcatNet(torch.nn.Module):
    def __init__(
        self, 
        args,
        GeminiMol_path = 'geminimol/GeminiMol_M5086'
        ):
        super((ConcatNet), self).__init__()
        self.GeminiMol_Net = GeminiMol(model_name=GeminiMol_path)
        del self.GeminiMol_Net.Decoder
        self.Integrate_Net = TestNet(
            Inner_dim = args['multi_head_attention_dim'],
            num_heads = args['multi_head_attention_head'],
            stack = 6,
            GeminiMol_path = 'geminimol/GeminiMol_M5086'
        )
        
    
    def forward(self, data_df, protein_embedding, device):
        protein_embedding = protein_embedding.to(device)
        out_distance_map, out_affinity, out_site = self.Integrate_Net(data_df, protein_embedding)
        return out_distance_map, out_affinity, out_site

    def get_last_shared_layer(self):
        return self.Integrate_Net.MHA_Cross_layer
    
class MTL_train(torch.nn.Module):
    def __init__(self, model, args, n_tasks=3):
        super(MTL_train, self).__init__()
        self.model = model
        self.weights = torch.nn.Parameter(torch.ones(n_tasks).float())
        # define loss functions
        self.contact_loss_fn = nn.MSELoss()
        self.site_focal_loss = focal_loss(alpha=[1, args['focal_loss_alpha']], gamma=args['focal_loss_gamma'], num_classes=2)
        self.affinity_loss_fn = nn.L1Loss()
        
    def forward(self, data_df, protein_embedding, device):
        pred_distance_map, pred_affinity, pred_site = self.model(data_df, protein_embedding, device)
        return (pred_distance_map, pred_site, pred_affinity)

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()


def inference(csv, device, model, checkpoint_name):
    #inference for one-protein/multiple ligands 
    csv_name = csv.replace('.csv','')
    df1 = pd.read_csv(csv)
    result_df = pd.DataFrame(columns=['index','affinity'])
    site_dict = {}
    distance_map_dict = {}
    affinity_dict = {}
    for index, content in tqdm(df1.iterrows()):
        seq = content['protein_seq']
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        protein_data = [("protein1", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(protein_data)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].to(device)
        input_df = df1[df1['index'] == index]
        if not input_df.empty:
            results = model(input_df, token_representations, device) #pred_distance_map, pred_site, pred_affinity
            site_dict[index] = results[1].detach().cpu()
            distance_map_dict[index] = results[0].detach().cpu()
            affinity_dict[index] = results[-1].detach().cpu()
            result_df.loc[len(result_df)] = [content['index'], results[-1].detach().cpu()]

    result_df.to_csv(f'{csv_name}_{checkpoint_name}_results.csv', index=False)
    np.save(f'{csv_name}_{checkpoint_name}_contacts.npy', distance_map_dict)
    np.save(f'{csv_name}_{checkpoint_name}_site.npy', site_dict)
    np.save(f'{csv_name}_{checkpoint_name}_affinity.npy', affinity_dict)
        

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
    ConcatenateNet = MTL_train(ConcatNet(args=args), args=args)
    ConcatenateNet.to(device)
    checkpoint_file = sys.argv[2]
    checkpoint = torch.load(checkpoint_file)
    ConcatenateNet.load_state_dict(checkpoint)
    list = []
    checkpoint_name = checkpoint_file.split('/')[-1].replace('.pt', '')
    csv = sys.argv[3]
    inference(csv, device, ConcatenateNet, checkpoint_name)
    