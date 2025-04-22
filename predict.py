from utils.get_ProtT5 import get_batch_protT5
from model.binding_network.binding_network_predict import binding_predict_function
from model.linker_network.linker_network_predict import linker_predict_function
import argparse
import os
import pandas as pd

BASE_PATH = os.getcwd()

parser = argparse.ArgumentParser(description="Run specific network based on input type")
parser.add_argument("-t", "--type_prediction", choices=["binding", "linker", "all"], required=True, help="Specify the prediction type:\n"
         "  binding: Represents disordered multiple binding functions.\n"
         "  linker: Represents disordered flexible linker functions.\n"
         "  all: Represents both of the above functions.")
parser.add_argument("-i", "--input_fasta", required=True, help="Path to the FASTA file")
parser.add_argument("-d", "--device", required=True, help="Specify the processor to use:\n"
         "  gpu: Use GPU for computation; append a number to indicate the GPU card index.(e.g., gpu0)\n"
         "  cpu: Use CPU for computation.")
args = parser.parse_args()

def fasta_to_dataframe2(file_name):
    acc_list = []
    seq_list = []
    with open(file_name,'r') as f:
        lines = f.readlines()
        for i in list(range(0, len(lines)-1, 2)):
            acc = lines[i].strip()[1:]  
            aa = lines[i+1].strip()
            acc_list.append(acc)
            seq_list.append(aa)     
    fasta_df = pd.DataFrame({'ID': acc_list, 'Sequence': seq_list})
    return fasta_df

def fasta_to_dataframe5(file_name):
    caid_acc = []
    caid_aa = []
    PB_list = []
    NB_list = []
    LB_list = []
    IB_list = []
    SB_list = []
    with open(file_name,'r') as f:
        lines = f.readlines()
        for i in list(range(0, len(lines)-6, 7)):
            acc = lines[i].strip()[1:]  
            aa = lines[i+1].strip()
            PB = lines[i+2].strip()
            NB = lines[i+3].strip()
            LB = lines[i+4].strip()
            IB = lines[i+5].strip()
            SB = lines[i+6].strip()
            
            caid_acc.append(acc)
            caid_aa.append(aa)
            PB_list.append(PB.split(','))
            NB_list.append(NB.split(','))
            LB_list.append(LB.split(','))
            IB_list.append(IB.split(','))
            SB_list.append(SB.split(','))
    caid_df = pd.DataFrame({'ID': caid_acc, 'Sequence': caid_aa,  'PB_scores': PB_list, 'NB_scores': NB_list, 'LB_scores': LB_list, 'IB_scores': IB_list, 'SB_scores': SB_list})
    return caid_df 
     
def fasta_to_dataframe3(file_name):
    caid_acc = []
    caid_aa = []
    DFL_list = []
    with open(file_name,'r') as f:
        lines = f.readlines()
        for i in list(range(0, len(lines)-2, 3)):
            acc = lines[i].strip()[1:]  
            aa = lines[i+1].strip()
            DFL = lines[i+2].strip()
            
            caid_acc.append(acc)
            caid_aa.append(aa)
            DFL_list.append(DFL.split(','))
    caid_df = pd.DataFrame({'ID': caid_acc, 'Sequence': caid_aa, 'linker_scores': DFL_list})
    return caid_df

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
fasta_path = os.path.join(BASE_PATH, args.input_fasta)
output_path = os.path.join(BASE_PATH, "data_save", os.path.splitext(args.input_fasta)[0])
create_folder(output_path)
#print(output_path)

protT5_save_path = os.path.join(output_path,"embedding_features/")
create_folder(protT5_save_path)
#print(protT5_save_path)

result_path = os.path.join(output_path,"result/")
create_folder(result_path)
#print(result_path)

protT5_model_path = os.path.join(BASE_PATH, "utils/prot_t5_xl_uniref50/")
seq_df = fasta_to_dataframe2(fasta_path)
device_selection = args.device
get_batch_protT5(seq_df, protT5_model_path, device_selection, protT5_save_path)


if args.type_prediction == "binding":
    binding_model_path = os.path.join(BASE_PATH,"model/binding_network/Q32_150.pth")
    binding_result_df = binding_predict_function(protT5_save_path, seq_df, device_selection, binding_model_path)
    with open(os.path.join(result_path,"binding_scores.txt"), 'w') as file:
        for index, row in binding_result_df.iterrows():
            file.write('>' + str(row['ID']) + '\n')
            file.write(str(row['Sequence']) + '\n')
            file.write(','.join(map(str, row['PB_scores'])) + '\n')
            file.write(','.join(map(str, row['NB_scores'])) + '\n')
            file.write(','.join(map(str, row['LB_scores'])) + '\n')
            file.write(','.join(map(str, row['IB_scores'])) + '\n')
            file.write(','.join(map(str, row['SB_scores'])) + '\n')
    with open(os.path.join(result_path,"binding_binary.txt"), 'w') as file:
        for index, row in binding_result_df.iterrows():
            file.write('>' + str(row['ID']) + '\n')
            file.write(str(row['Sequence']) + '\n')
            file.write(','.join(map(str, row['PB_binary'])) + '\n')
            file.write(','.join(map(str, row['NB_binary'])) + '\n')
            file.write(','.join(map(str, row['LB_binary'])) + '\n')
            file.write(','.join(map(str, row['IB_binary'])) + '\n')
            file.write(','.join(map(str, row['SB_binary'])) + '\n')

elif args.type_prediction == "linker":
    linker_model_path = os.path.join(BASE_PATH,"model/linker_network/Q1_400.pth")
    linker_result_df = linker_predict_function(protT5_save_path, seq_df, device_selection, linker_model_path) 
    with open(os.path.join(result_path,"linker_scores.txt"), 'w') as file:
        for index, row in linker_result_df.iterrows():
            file.write('>' + str(row['ID']) + '\n')
            file.write(str(row['Sequence']) + '\n')
            file.write(','.join(map(str, row['linker_scores'])) + '\n')
    with open(os.path.join(result_path,"linker_binary.txt"), 'w') as file:
        for index, row in linker_result_df.iterrows():
            file.write('>' + str(row['ID']) + '\n')
            file.write(str(row['Sequence']) + '\n')
            file.write(','.join(map(str, row['linker_binary'])) + '\n')
    
elif args.type_prediction == "all":
    if not os.path.exists(os.path.join(result_path,"binding_scores.txt")):
        binding_model_path = os.path.join(BASE_PATH,"model/binding_network/Q32_150.pth")
        binding_result_df = binding_predict_function(protT5_save_path, seq_df, device_selection, binding_model_path)
    else:
        binding_score_df = fasta_to_dataframe5(os.path.join(result_path,"binding_scores.txt"))
        binding_binary_df = fasta_to_dataframe5(os.path.join(result_path,"binding_binary.txt"))[['ID', 'PB_scores', 'NB_scores', 'LB_scores', 'IB_scores', 'SB_scores']].rename(columns={
        'PB_scores': 'PB_binary',
        'NB_scores': 'NB_binary',
        'LB_scores': 'LB_binary',
        'IB_scores': 'IB_binary',
        'SB_scores': 'SB_binary'
    })
        binding_result_df = pd.merge(binding_score_df, binding_binary_df, on='ID', how='inner')
        #print(binding_result_df)
    if not os.path.exists(os.path.join(result_path,"linker_scores.txt")):
        linker_model_path = os.path.join(BASE_PATH,"model/linker_network/Q1_400.pth")
        linker_result_df = linker_predict_function(protT5_save_path, seq_df, device_selection, linker_model_path)
    else:
        linker_score_df = fasta_to_dataframe3(os.path.join(result_path,"linker_scores.txt"))
        linker_binary_df = fasta_to_dataframe3(os.path.join(result_path,"linker_binary.txt"))[['ID','linker_scores']].rename(columns={'linker_scores': 'linker_binary'})
        linker_result_df = pd.merge(linker_score_df, linker_binary_df, on='ID', how='inner')
        #print(linker_result_df)
    with open(os.path.join(result_path,"all_scores.txt"), 'w') as file:
        for index, row in binding_result_df.iterrows():
            file.write('>' + str(row['ID']) + '\n')
            file.write(str(row['Sequence']) + '\n')
            file.write(','.join(map(str, row['PB_scores'])) + '\n')
            file.write(','.join(map(str, row['NB_scores'])) + '\n')
            file.write(','.join(map(str, row['LB_scores'])) + '\n')
            file.write(','.join(map(str, row['IB_scores'])) + '\n')
            file.write(','.join(map(str, row['SB_scores'])) + '\n')
            file.write(','.join(map(str, linker_result_df[linker_result_df["ID"]==row["ID"]]['linker_scores'].values[0])) + '\n')
    with open(os.path.join(result_path,"all_binary.txt"), 'w') as file:
        for index, row in binding_result_df.iterrows():
            file.write('>' + str(row['ID']) + '\n')
            file.write(str(row['Sequence']) + '\n')
            file.write(','.join(map(str, row['PB_binary'])) + '\n')
            file.write(','.join(map(str, row['NB_binary'])) + '\n')
            file.write(','.join(map(str, row['LB_binary'])) + '\n')
            file.write(','.join(map(str, row['IB_binary'])) + '\n')
            file.write(','.join(map(str, row['SB_binary'])) + '\n')
            file.write(','.join(map(str, linker_result_df[linker_result_df["ID"]==row["ID"]]['linker_binary'].values[0])) + '\n')
else:
    print("Invalid input type. Please specify one of the following options: 'binding', 'linker', or 'all'.")
    
    
    
    
    



