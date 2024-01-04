## Project Files Overview

### Python Code
- `main.py` - Code for initializing everything and running the code (change print flags (0,1) for relevant parameters).
- `dataload.py` - Code to load train, validation, test data from raw audio or preprocessed spectrograms.
- `models.py` - Code for all the models used.
- `train.py` - Code to train the model based on argument parameters.
- `misc.py` - Helper functions to load JSON, write summary logs.
- `splitter.py` - Splits the .pkl file in the dataset into train (parts 0-9, a, b), validation (part c), and test (parts d, e, f). Samples were split manually.

### JSON Files
- `datasets.json` - Contains the locations to the dataset paths for BC4 and local.
- `labels.json` - Contains the first 50 labels used for classification.
- `min_max_values.json` - Contains the min and max values from the raw audio dataset.

### Bash Scripts
- `main.sh` - Slurm script used to submit job to BC4.
  - Arguments to run the main function (only relevant args with defaults included):
    - `--sgd-momentum 0.93`
    - `--learning-rate 0.2`
    - `--epochs 20`
    - `--model ChunkResCNN`
    - `--conv-length 256`
    - `--conv-stride 256`
    - `--weight-decay 1e-5`

### Logs 
- `bc4_outs` - All logs mentioned in the report are included here.

## Files Not Included

### Python Code (for local)
- `spec.py` - Code to preprocess the raw audio data into mel spectrograms.

### BC4 datasets (mnt/storage/scratch/ee20947/MagnaTagATune)
- `annotations` - `train.pkl`, `val.pkl`, `test.pkl` files are the same for both raw audio and spectrograms.
- `samples` - Contains `.npy` files for train, validation, and test raw audio samples under their respective names.
- `spectrograms` - Contains the `.npy` files for all the different spectrograms tested. Formats include:
  - `..._1024_128`
  - `..._1024_64`
  - `..._512_128`
  - `..._512_64`
  - `...Chunked_1024_128`
  - `...Chunked_512_128`
  - The first and second numbers denote `n_fft` and `n_mels` used for creating the spectrograms.
  - `...Chunked...` spectrograms are chunked to the size of 3.69 seconds.
  - Related paper: [https://arxiv.org/abs/2006.00751v1](https://arxiv.org/abs/2006.00751v1)
- Dataset link: [https://uob-my.sharepoint.com/:f:/g/personal/ee20947_bristol_ac_uk/EgmnovNnrQRDgrPXiyjl-04BIL0acybysBnD3-xQoZiZfQ](https://uob-my.sharepoint.com/:f:/g/personal/ee20947_bristol_ac_uk/EgmnovNnrQRDgrPXiyjl-04BIL0acybysBnD3-xQoZiZfQ)

## Run Instructions
1. Open `main.sh`.
2. Change arguments at the bottom:
   - Use `BaseCNN` for replication of the paper by Sander et al. or `ChunkResCNN1` for extension.
   - All other arguments can be left as default.
3. Submit the job to BC4 with the command: `sbatch main.sh`
