import torch
from transformers import T5Tokenizer, T5Model, T5EncoderModel
import re
import os
import numpy as np
import gc
import pandas as pd
from transformers import logging
import argparse
logging.set_verbosity_warning()
logging.set_verbosity_error()

def get_protT5(seq,model_path,device_selection):
    seq_lists = [seq]
    seq_lists = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq_lists]
    
    if device_selection.startswith("gpu"):
        gpu_index = device_selection[3:]  
        device = torch.device(f'cuda:{gpu_index}') 
    else:
        device = torch.device("cpu")
        
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
  
    model = T5EncoderModel.from_pretrained(model_path)
    #print(f"Model loaded on device: {next(model.parameters()).device}")
    gc.collect()
    
    model = model.to(device)
    #print(f"Model loaded on device: {next(model.parameters()).device}")
    model = model.eval()
    
    ids = tokenizer.batch_encode_plus(seq_lists, add_special_tokens=True, padding='longest')
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
  	    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
           
    embedding = embedding.cpu().numpy()
    embedding = embedding[0][1:]
    
    return embedding
  

def get_batch_protT5(df, model_path, device_selection, save_path):
    for index, row in df.iterrows():
        ID = row['ID']
        seq = row['Sequence']
        ID_protT5_path = os.path.join(save_path, f'{ID}.npy')
        if not os.path.exists(ID_protT5_path):
            ID_embedding = get_protT5(seq,model_path, device_selection)
            np.save(ID_protT5_path ,ID_embedding)
    

def fasta_to_dataframe2(file_name):
    caid_acc = []
    caid_aa = []
    with open(file_name,'r') as f:
        lines = f.readlines()
        for i in list(range(0, len(lines)-1, 2)):
            acc = lines[i].strip()[1:]  
            aa = lines[i+1].strip()
            caid_acc.append(acc)
            caid_aa.append(aa)     
    caid_df = pd.DataFrame({'ID': caid_acc, 'Sequence': caid_aa})
    return caid_df


def save_ProtT5_embedding(fasta_file,model_path, device_selection, save_path):
    df = fasta_to_dataframe2(fasta_file)
    get_batch_protT5(df, model_path, device_selection, save_path)
    


