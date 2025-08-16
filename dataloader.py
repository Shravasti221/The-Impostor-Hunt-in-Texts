import os
import pandas as pd

def read_texts_from_dir(dir_path):
    data = []
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
                    text1 = f1.read().strip()
                with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
                    text2 = f2.read().strip()
                index = int(folder_name.split('_')[-1])
                data.append((index, text1, text2))
            except Exception:
                continue
    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')

def load_dataset(train_path, label_path):
    df_train = read_texts_from_dir(train_path).reset_index()
    df_train_gt = pd.read_csv(label_path)
    df = df_train.merge(df_train_gt, on='id', how='left')

    def split_real_fake(row):
        if row['real_text_id'] == 1:
            return pd.Series([row['file_1'], row['file_2']])
        else:
            return pd.Series([row['file_2'], row['file_1']])

    df[['real_text', 'fake_text']] = df.apply(split_real_fake, axis=1)
    return df[['id', 'real_text', 'fake_text']]
