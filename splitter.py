import pandas as pd
import pickle
import os

def split(dataset_path):

    #load the data
    df = pd.read_pickle(dataset_path)

    
    #split the df based on 'part' column being 'c'
    df_c = df[df['part'] == 'c']
    df_not_c = df[df['part'] != 'c']

    #the start of filepath needs to be changed
    df_c['file_path'] = df_c['file_path'].str.replace(' train/c', 'val/c')
    #df['file_path'] = df['file_path'].str.replace('val/', 'test/')

    print("\n'part' is 'c':")
    print(df_c.head())
    print(df_c.tail())
    print(df_c.info())
    print(df_c.describe())

    print("\n'part' is not 'c':")
    print(df_not_c.head())
    print(df_not_c.tail())
    print(df_not_c.info())
    print(df_not_c.describe())

    #save df as train and val
    df_c.to_pickle('train.pkl')
    df_not_c.to_pickle('val.pkl')
    
    #df.to_pickle('TestNew.pkl')
    
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.describe())

    #return df
    return df_c, df_not_c


#uncomment to split the pickles
'''
val = split('/Users/vasudevmenon/ADL/cw/TestNew.pkl')

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

print(val)
'''

def assign_labels_to_chunks(base_directory, label_dataframe_path):
    # Load the original DataFrame
    label_dataframe = pd.read_pickle(label_dataframe_path)

    updated_data = []

    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.npy'):
                    # Construct the new file path for the chunk
                    chunk_path = os.path.join(subdir_path, file)

                    # Extract the original filename from the chunk filename
                    original_filename, _ = file.rsplit('_chunk_', 1)
                    original_filename += '.npy'

                    # Find the matching row in the original DataFrame
                    matching_rows = label_dataframe[label_dataframe['file_path'].str.contains(original_filename)]

                    if not matching_rows.empty:
                        for _, row in matching_rows.iterrows():
                            updated_data.append({
                                'file_path': chunk_path,
                                'label': row['label'],
                                'part': row['part']
                            })
                    else:
                        print(f"Label not found for {original_filename}")

    # Create a new DataFrame with the updated data
    updated_df = pd.DataFrame(updated_data, columns=label_dataframe.columns)

    return updated_df

# Example usage
base_directory = '/mnt/storage/scratch/ee20947/MagnaTagATune/spectrograms/specPreprocessedChunked_512_128/train'  # Replace with your base directory path
label_dataframe_path = '/mnt/storage/scratch/ee20947/MagnaTagATune/annotations/train.pkl'   # Path to your original DataFrame .pkl file
updated_df = assign_labels_to_chunks(base_directory, label_dataframe_path)

# Save the updated DataFrame as a new .pkl file
updated_df.to_pickle('/mnt/storage/scratch/ee20947/MagnaTagATune/spec_annotations/trainSpecChunk.pkl')
