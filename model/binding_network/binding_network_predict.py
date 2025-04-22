import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class Binding_dataset(Dataset):
    def __init__(self,prot5_path,all_df):
        self.all_df = all_df
        self.prot5_path = prot5_path
            
    def __len__(self):
        self.len = len(self.all_df["ID"].tolist())
        return self.len

    def __getitem__(self, index):
        data_list = self.all_df["ID"].tolist()
        ID = data_list[index]
        all_df = self.all_df.set_index('ID')
        seq = all_df.loc[ID,'Sequence']
        id_path = self.prot5_path + ID + '.npy'
        data = np.load(id_path)
        seq_embedding = torch.tensor(data, dtype=torch.float)
        return ID,seq,seq_embedding
 
        
def binding_predict_function(prot5_path, all_df, device_selection, model_path):
    test_dataset = Binding_dataset(prot5_path, all_df)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)
    if device_selection.startswith("gpu"):
        gpu_index = gpu_index = device_selection[3:]  
        device = torch.device(f'cuda:{gpu_index}') 
    else:
        device = torch.device("cpu")
    model = torch.load(model_path,map_location = torch.device('cpu'))
    #print(f"Model loaded on device: {next(model.parameters()).device}")
    model.to(device)
    #print(f"Model loaded on device: {next(model.parameters()).device}")
    #print(f"Current device: {device}")
    all_PB_scores = []
    all_NB_scores = []
    all_LB_scores = []
    all_IB_scores = []
    all_SB_scores = []
    
    all_PB_binary = []
    all_NB_binary = []
    all_LB_binary = []
    all_IB_binary = []
    all_SB_binary = []
    
    ID_list = []
    seq_list = []

    model.eval()
    with torch.no_grad():
        for i, loader in enumerate(test_dataloader):
            ID, seq, feature  = loader
            ID_list.append(ID[0])
            seq_list.append(seq[0])

            feature = feature.to(device)
            outputs = model(feature)
            PB_score, NB_score, LB_score, IB_score, SB_score = [outputs[:, :, i].unsqueeze(-1) for i in
                                                                range(outputs.shape[2])]
            PB_score = [round(score, 4) for score in PB_score.cpu().flatten().tolist()]
            NB_score = [round(score, 4) for score in NB_score.cpu().flatten().tolist()]
            LB_score = [round(score, 4) for score in LB_score.cpu().flatten().tolist()]
            IB_score = [round(score, 4) for score in IB_score.cpu().flatten().tolist()]
            SB_score = [round(score, 4) for score in SB_score.cpu().flatten().tolist()]

            all_PB_scores.append(PB_score)
            all_NB_scores.append(NB_score)
            all_LB_scores.append(LB_score)
            all_IB_scores.append(IB_score)
            all_SB_scores.append(SB_score)
            
            all_PB_binary.append([1 if score >= 0.284 else 0 for score in PB_score])
            all_NB_binary.append([1 if score >= 0.069 else 0 for score in NB_score])
            all_LB_binary.append([1 if score >= 0.127 else 0 for score in LB_score])
            all_IB_binary.append([1 if score >= 0.094 else 0 for score in IB_score])
            all_SB_binary.append([1 if score >= 0.035 else 0 for score in SB_score])
            
            
    data_dict = {
        'ID': ID_list,
        'Sequence': seq_list,
        'PB_scores': all_PB_scores,
        'NB_scores': all_NB_scores,
        'LB_scores': all_LB_scores,
        'IB_scores': all_IB_scores,
        'SB_scores': all_SB_scores,
        'PB_binary': all_PB_binary,
        'NB_binary': all_NB_binary,
        'LB_binary': all_LB_binary,
        'IB_binary': all_IB_binary,
        'SB_binary': all_SB_binary,
        }
    df = pd.DataFrame(data_dict)
    return df
    

    