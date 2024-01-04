**Files included here are:**

Python Code
 - main.py - Code for initialising everything and running the code (change print flags (0,1) for relevant parameters)
 - dataload.py - Code to load train, val, test data from raw audio or preprocessed spectrograms
 - models.py - Code for all the models used
 - train.py - Code to train the model based on arg parameters
 - misc.py - Code with helper functions to load json, write summary logs
 - splitter.py - Code to split the .pkl file in the dataset into train(parts 0,1,2,3,4,5,6,7,8,9,a,b), val(part c), test(parts d,e,f)
               - Samples were split manually

JSON Files
 - datasets.json - Contains the locations to the dataset paths for BC4 and local
 - labels.json - Contains the first 50 labels used for classification
 - min_max_values.json - Contains the min and max values from the raw audio dataset

 Bash Scripts
 - main.sh - Slurm script to used to submit job to BC4
           - Use the below arguments can be used to run the main function. Note: Only relevant args with defaults included below.
                --sgd-momentum 0.93
                --learning-rate 0.2
                --epochs 20
                --model ChunkResCNN
                --conv-length 256
                --conv-stride 256
                --weight-decay 1e-5

Logs 
 - bc4_outs - All logs that have been mentioned in the report have been included here
    

**Files not included here:**

Python Code (for local)
 - spec.py - Code to preprocess the raw audio data into mel spectrograms

BC4 datasets (mnt/storage/scratch/ee20947/MagnaTagATune)
 - annotations - train.pkl, val.pkl, test.pkl files are the same for both raw audio and spectrograms
 - samples - Contains .npy files for train, val and test raw audio samples under their respective names
 - spectrograms - Contains the .npy files for all the different spectrograms tested. The below follow specPreprocessed...
                ..._1024_128
                ..._1024_64
                ..._512_128
                ..._512_64
                ...Chunked_1024_128
                ...Chunked_512_128
                - Each of the above contains the spectrogram files, where the first and second number are the n_fft and n_mels used for the creation of the spectrograms
                - The ones with ...Chunked... have been chunked to the size of 3.69 in relation to the paper (https://arxiv.org/abs/2006.00751v1)
Please find all datasets here - https://uob-my.sharepoint.com/:f:/g/personal/ee20947_bristol_ac_uk/EgmnovNnrQRDgrPXiyjl-04BIL0acybysBnD3-xQoZiZfQ

**RUN INSTRUCTIONS**
- Open main.sh
- Change arguments at the bottom
    - Use BaseCNN for replication of paper by Sander et al. or ChunkResCNN1 for extension
    - All other arguments can be default
- Submit job to BC4 with the command: sbatch main.sh