import os
import numpy as np
import torch
import torchaudio.transforms as transforms


def preprocess(base_dataset_path, base_save_path, sample_rate=16000, n_fft=1024, f_min=0.0, f_max=8000.0, n_mels=128, chunk_duration=3.69):
    '''
    Creation of mel spectrograms done on local and uploaded to scratch on bc4 (raw audio npy -> spectrogram npy)

    BC4 scratch will have chunked and non-chunked versions of the spectrograms
    '''
    spec_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels)
    amp = transforms.AmplitudeToDB()
    chunk_size = int(sample_rate * chunk_duration)

    for subdir in os.listdir(base_dataset_path):
        subdir_path = os.path.join(base_dataset_path, subdir)
        if os.path.isdir(subdir_path):

            save_subdir_path = os.path.join(base_save_path, subdir)
            os.makedirs(save_subdir_path, exist_ok=True)

            for filename in os.listdir(subdir_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(subdir_path, filename)

                    waveform = np.load(file_path)
                    tensor = torch.from_numpy(waveform).float()

                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)

                    num_chunks = int(np.ceil(tensor.shape[1] / chunk_size))
                    for i in range(num_chunks):
                        start_sample = i * chunk_size
                        end_sample = start_sample + chunk_size
                        chunk_tensor = tensor[:, start_sample:end_sample]

                        #pad last chunk
                        if chunk_tensor.shape[1] < chunk_size:
                            padding_size = chunk_size - chunk_tensor.shape[1]
                            chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, padding_size))

                        try:
                            spec = spec_transform(chunk_tensor)
                            spec_db = amp(spec)
                            save_file_name = f"{filename[:-4]}_chunk_{i}.npy"
                            save_file_path = os.path.join(save_subdir_path, save_file_name)
                            np.save(save_file_path, spec_db.numpy())

                        except Exception as e:
                            print(f"Failed on chunk {i} of {filename}: {e}")


#uncomment to run locally
'''
base_dataset_path = '/Users/vasudevmenon/ADL/cw/MagnaTagATune/samples/test'
base_save_path = '/Users/vasudevmenon/ADL/cw/MagnaTagATune/samples/specPreprocessedChunked/test'
preprocess_and_save_npy(base_dataset_path, base_save_path)
'''