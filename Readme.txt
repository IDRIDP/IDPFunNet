Installation and implementation of IDPFunNet


1 Description
    IDPFunNet(A novel deep learning framework for systematic prediction of intrinsically disordered protein functions using protein language model) is a software designed for the accurate prediction of various disordered binding functions and disordered flexible linker functions of proteins. Due to the different characteristics of disordered binding functions and flexible linker functions, these two categories are predicted separately. Using sequence semantic vectors obtained from the protein language model ProtT5, a multi-task learning framework and multi-scale information fusion network are employed to predict disordered functional annotations for protein binding, nucleic acid binding, lipid binding, ion binding, and small molecule binding. A global information network is used to predict the disordered functional annotations of flexible linker.
    
    
2 Installation
    2.1 Install Anaconda, the anaconda can be downloaded from https://www.anaconda.com/
    2.2 Download the protein language model ProtT5 network weights file (https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/pytorch_model.bin?download=true) and copy it to the `"/IDPFunNet/utils/prot_t5_xl_uniref50/"` folder.
    2.3 Navigate to the installation path: cd IDPFunNet
    2.4 Create a new environment with conda and configure it: conda env create -f IDPFunNet.yml
    2.5 Activate environment: conda activate IDPFunNet
   
    
3 Run IDPFunNet
    3.1 Predict disordered multiple binding functions: 
        (1) Command: python predict.py -t binding -i example.fasta -d gpu0
        (2) Result: The files `binding_scores.txt` and `binding_binary.txt` are saved in the directory `"/IDPFunNet/data_save/example/result/"`. `binding_scores.txt` contains the predicted scores, and `binding_binary.txt` contains the                binary classification results for disordered binding functions of each sequence.
        (3) File Contents:
              Line 1: >Sequence ID
              Line 2: Protein sequence (1-letter amino acid encoding)
              Line 3: Predicted results for disordered protein-binding functions
              Line 4: Predicted results for disordered nucleic acid-binding functions
              Line 5: Predicted results for disordered lipid-binding functions
              Line 6: Predicted results for disordered ion-binding functions
              Line 7: Predicted results for disordered small molecule-binding functions
              
    3.2 Predict disordered flexible linker functions:
        (1) Command: python predict.py -t linker -i example.fasta -d gpu0
        (2) Result: The files `linker_scores.txt` and `linker_binary.txt` are saved in the directory `"/IDPFunNet/data_save/example/result/"`. `linker_scores.txt` contains the predicted scores, and `linker_binary.txt` contains the binary             classification results for disordered flexible linker functions of each sequence.
        (3) File Contents:
              Line 1: >Sequence ID
              Line 2: Protein sequence (1-letter amino acid encoding)
              Line 3: Predicted results for disordered flexible linker functions
              
    3.3 Predict disordered multiple binding and flexible linker functions:
        (1) Command: python predict.py -t all -i example.fasta -d gpu0
        (2) Result: The files `all_scores.txt` and `all_binary.txt` are saved in the directory `"/IDPFunNet/data_save/example/result/"`. `all_scores.txt` contains the predicted scores, and `all_binary.txt` contains the binary                         classification results for disordered functions of each sequence.
        (3) File Contents:
              Line 1: >Sequence ID
              Line 2: Protein sequence (1-letter amino acid encoding)
              Line 3: Predicted results for disordered protein-binding functions
              Line 4: Predicted results for disordered nucleic acid-binding functions
              Line 5: Predicted results for disordered lipid-binding functions
              Line 6: Predicted results for disordered ion-binding functions
              Line 7: Predicted results for disordered small molecule-binding functions
              Line 8: Predicted results for disordered flexible linker functions
    
    3.4 Explanation of some parameters
        (1) -t specifies the prediction type, with three options: binding, linker, and all, representing disordered multiple binding functions, disordered flexible linker functions, or both, respectively.
        (2) -i specifies the input FASTA file.
        (3) -d specifies the processor to use, allowing you to choose between gpu or cpu. If you select gpu, you need to append a number to indicate the GPU card index(e.g., gpu0).
    