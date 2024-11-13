#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from packaging.version import Version as LV
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import sys
import numpy as np

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert LV(torch.__version__) >= LV("1.0.0")


def correct(output, target):
    predicted = output.argmax(1)
    correct_ones = (predicted == target).type(torch.float)
    return correct_ones.sum().item()


def train(data_loader, model, scheduler, optimizer):
    model.train()

    num_batches = 0
    num_items = 0

    total_loss = 0
    total_correct = 0
    for input_ids, input_mask, labels in data_loader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)

        output = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        loss = output[0]
        logits = output[1]

        total_loss += loss
        num_batches += 1

        total_correct += correct(logits, labels)
        num_items += len(labels)

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {'loss': total_loss / num_batches, 'accuracy': total_correct / num_items}


def test(test_loader, model):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for input_ids, input_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)

            output = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            logits = output[0]
            total_correct += correct(logits, labels)

    return {'loss': test_loss / num_batches, 'accuracy': total_correct / num_items}


def log_measures(ret, log, prefix, epoch):
    if log is not None:
        for key, value in ret.items():
            log.add_scalar(f"{prefix}_{key}", value, epoch)


def main():
    try:
        import tensorboardX
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logdir = os.path.join(os.getcwd(), "logs", "20ng-bert-" + time_str)
        os.makedirs(logdir)
        log = tensorboardX.SummaryWriter(logdir)
    except (ImportError, FileExistsError):
        log = None

    datapath = os.getenv('DATADIR')
    if datapath is None:
        print("Please set DATADIR environment variable!")
        sys.exit(1)

    text_data_dir = os.path.join(datapath, "20_newsgroup")

    # Load data and labels
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                    labels.append(label_id)

    # Split the data into a training set and a test set
    TEST_SET = 4000
    (sentences_train, sentences_test,
     labels_train, labels_test) = train_test_split(texts, labels, test_size=TEST_SET, shuffle=True, random_state=42)

    # Prepare sentences with [CLS] token
    sentences_train = ["[CLS] " + s for s in sentences_train]
    sentences_test = ["[CLS] " + s for s in sentences_test]

    # Initialize BERT tokenizer
    BERTMODEL = 'bert-base-uncased'
    CACHE_DIR = os.path.join(datapath, 'transformers-cache')
    tokenizer = BertTokenizer.from_pretrained(BERTMODEL, cache_dir=CACHE_DIR, do_lower_case=True)

    # Tokenize sentences
    MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512
    tokenized_train = [tokenizer.tokenize(s)[:MAX_LEN_TRAIN - 1] + ['[SEP]'] for s in sentences_train]
    tokenized_test = [tokenizer.tokenize(s)[:MAX_LEN_TEST - 1] + ['[SEP]'] for s in sentences_test]

    # Convert tokens to IDs and pad sequences
    ids_train = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_train]
    ids_train = np.array([np.pad(i, (0, MAX_LEN_TRAIN - len(i)), mode='constant') for i in ids_train])

    ids_test = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_test]
    ids_test = np.array([np.pad(i, (0, MAX_LEN_TEST - len(i)), mode='constant') for i in ids_test])

    # Attention masks
    amasks_train = [[float(i > 0) for i in seq] for seq in ids_train]
    amasks_test = [[float(i > 0) for i in seq] for seq in ids_test]

    # Split train data into training and validation sets
    (train_inputs, validation_inputs, train_labels, validation_labels) = train_test_split(ids_train, labels_train, random_state=42, test_size=0.1)
    (train_masks, validation_masks, _, _) = train_test_split(amasks_train, ids_train, random_state=42, test_size=0.1)

    # Convert to tensors
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    test_inputs = torch.tensor(ids_test)
    test_labels = torch.tensor(labels_test)
    test_masks = torch.tensor(amasks_test)

    # Create DataLoaders
    BATCH_SIZE = 16
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    validation_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_dataset)
    validation_loader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=BATCH_SIZE)

    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    # Initialize BERT model for classification
    model = BertForSequenceClassification.from_pretrained(BERTMODEL, cache_dir=CACHE_DIR, num_labels=20)
    model = model.to(device)

    num_epochs = 6
    weight_decay = 0.01
    lr = 2e-5
    warmup_steps = int(0.1 * len(train_loader) * num_epochs)

    # Layer-wise learning rate decay (LLRD)
    layers = model.bert.encoder.layer
    lr_multipliers = [lr / (2.6 ** i) for i in range(len(layers))]

    optimizer_grouped_parameters = []
    for idx, layer in enumerate(layers):
        params = {'params': layer.parameters(), 'lr': lr_multipliers[idx], 'weight_decay': weight_decay}
        optimizer_grouped_parameters.append(params)

    optimizer_grouped_parameters.append({'params': model.classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay})

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * num_epochs)

    start_time = datetime.now()
    for epoch in range(num_epochs):
        train_ret = train(train_loader, model, scheduler, optimizer)
        log_measures(train_ret, log, "train", epoch)

        val_ret = test(validation_loader, model)
        log_measures(val_ret, log, "val", epoch)
        print(f"Epoch {epoch+1}: train loss: {train_ret['loss']:.6f} train accuracy: {train_ret['accuracy']:.2%}, val accuracy: {val_ret['accuracy']:.2%}")

    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    # Final evaluation on test dataset
    ret = test(test_loader, model)
    print(f"\nTesting: accuracy: {ret['accuracy']:.2%}")


if __name__ == "__main__":
    main()