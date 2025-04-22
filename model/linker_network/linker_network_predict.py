import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class linker_dataset(Dataset):
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
 
        
def linker_predict_function(prot5_path, all_df, device_selection, model_path):
    test_dataset = linker_dataset(prot5_path, all_df)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)
    if device_selection.startswith("gpu"):
        gpu_index = device_selection[3:]  
        device = torch.device(f'cuda:{gpu_index}') 
    else:
        device = torch.device("cpu")
    model = torch.load(model_path,map_location = torch.device('cpu'))
    #print(f"Model loaded on device: {next(model.parameters()).device}")
    model.to(device)
    #print(f"Model loaded on device: {next(model.parameters()).device}")
    #print(f"Current device: {device}")
    
    all_linker_scores = []
    all_linker_binary = []
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
            linker_score = [outputs[:, :, i].unsqueeze(-1) for i in range(outputs.shape[2])][0]
            linker_score = [round(score, 4) for score in linker_score.cpu().flatten().tolist()]
            all_linker_scores.append(linker_score)
            all_linker_binary.append([1 if score >= 0.086 else 0 for score in linker_score])
            
    data_dict = {
        'ID': ID_list,
        'Sequence': seq_list,
        'linker_scores': all_linker_scores,
        'linker_binary': all_linker_binary,
        }
    df = pd.DataFrame(data_dict)
    return df
    
