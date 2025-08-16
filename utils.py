from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

def compare_lengths(df):
    df['real_len'] = df['real_text'].str.split().str.len()
    df['fake_len'] = df['fake_text'].str.split().str.len()
    count_real_longer = (df['real_len'] > df['fake_len']).sum()
    print(f"Real text is longer than fake text in {count_real_longer} out of {len(df)} cases.")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def make_hf_datasets(df, seed=42):
    dataset = Dataset.from_pandas(df)
    split = dataset.train_test_split(test_size=0.1, seed=seed)
    train_val = split['train'].train_test_split(test_size=0.125, seed=seed)
    return train_val['train'], train_val['test'], split['test']


import pandas as pd

def chunk_text(text, chunk_size=512, overlap=256):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    if len(words) <= chunk_size:
        chunks.append(' '.join(words))
    else:
        for start in range(0, len(words) - overlap, step):
            end = start + chunk_size
            chunks.append(' '.join(words[start:end]))
            if end >= len(words):
                break
    return chunks

def create_chunked_dataframe(df):
    texts = df['real_text'].tolist() + df['fake_text'].tolist()
    labels = [1] * len(df['real_text']) + [0] * len(df['fake_text'])

    chunked_texts, chunked_labels = [], []
    for text, label in zip(texts, labels):
        chunks = chunk_text(text)
        chunked_texts.extend(chunks)
        chunked_labels.extend([label] * len(chunks))

    return pd.DataFrame({'text': chunked_texts, 'label': chunked_labels})
