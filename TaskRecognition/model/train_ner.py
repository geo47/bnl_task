import argparse

import nltk
from datasets import load_from_disk

from TaskRecognition.model.callbacks import LoggingCallback
from TaskRecognition.model.entity_dataset import EntityDataset
from TaskRecognition.model.model import T5FineTuner
from TaskRecognition.model.util import set_seed

nltk.download('punkt')

import pytorch_lightning as pl


from transformers import (
    AutoTokenizer
)


set_seed(42)
dataset = load_from_disk("../dataset/entities_ds/ner_dataset")

args_dict = dict(
    data_dir="../dataset/entities_ds/ner_dataset",
    output_dir="../../models/",
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=40,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=1,
    num_train_epochs=60,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=True,
    opt_level='O1',
    max_grad_norm=1,
    seed=42,
)

args = argparse.Namespace(**args_dict)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
# print(tokenizer)

dataset_train = EntityDataset(
    tokenizer=tokenizer, dataset=dataset, type_path='train', max_len=args.max_seq_length)
dataset_valid = EntityDataset(
    tokenizer=tokenizer, dataset=dataset, type_path='valid', max_len=args.max_seq_length)

print(len(dataset_train))
d_train = dataset_train[0]
print(tokenizer.decode(d_train["source_ids"], skip_special_tokens=False))
print(tokenizer.decode(d_train["target_ids"], skip_special_tokens=False))

d_valid = dataset_valid[0]
print(tokenizer.decode(d_valid["source_ids"], skip_special_tokens=False))
print(tokenizer.decode(d_valid["target_ids"], skip_special_tokens=False))

model = T5FineTuner(args, ds_train=dataset_train, ds_valid=dataset_valid)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filename=args.output_dir+"/checkpoint.pth", monitor="val_loss", mode="min", save_top_k=1, verbose=True
)

train_params = dict(
    weights_save_path=args.output_dir,
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    #early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    #amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

trainer = pl.Trainer(**train_params)
trainer.fit(model)