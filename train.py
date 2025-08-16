import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from utils import compute_metrics

def train_model(train_dataset, val_dataset, model_name="google/electra-small-discriminator", num_epochs=30, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir="./electra_finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer
