import math
import torch.optim as optim
import transformers
import torch.nn as nn
from torchtext.legacy import data
import csv
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased')

pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
max_input_length = 256


def split_and_cut(sentence):
    tokens = sentence.strip().split(" ")
    tokens = tokens[:max_input_length]
    return tokens


def combine_mask_array(mask):
    mask = [str(m) for m in mask]
    return " ".join(mask)


# input data
train, dev, test = [], [], []

with open('./data/pnli_train.csv', encoding='utf-8') as fp:
    csvreader = csv.reader(fp)
    for x in csvreader:
        # x[2] will be the label (0 or 1). x[0] and x[1] will be the sentence pairs.
        train.append(x)

with open('./data/pnli_dev.csv', encoding='utf-8') as fp:
    csvreader = csv.reader(fp)
    for x in csvreader:
        # x[2] will be the label (0 or 1). x[0] and x[1] will be the sentence pairs.
        dev.append(x)

with open('./data/pnli_test_unlabeled.csv', encoding='utf-8') as fp:
    csvreader = csv.reader(fp)
    for x in csvreader:
        # x[0] and x[1] will be the sentence pairs.
        test.append(x)

# -----------loading data------------------------
train_df = []
test_df = []
train_length = len(train)
split = int(0.9 * train_length)
for i in range(train_length):
    curr_data = {}
    curr_data['label'] = train[i][2]
    sent_1 = '[CLS] ' + train[i][0] + ' [SEP] '
    sent_2 = train[i][1] + ' [SEP]'
    sent_1_token = tokenizer.tokenize(sent_1)
    sent_2_token = tokenizer.tokenize(sent_2)
    curr_data['sequence'] = " ".join(sent_1_token + sent_2_token)
    curr_data['attention_mask'] = combine_mask_array(
        [1] * len(sent_1_token + sent_2_token))
    curr_data['token_type'] = combine_mask_array([0] * len(
        sent_1_token) + [1] * len(sent_2_token))
    if i < split:
        train_df.append(curr_data)
    else:
        test_df.append(curr_data)

print('length of train df: ', len(train_df))
print('length of test df: ', len(test_df))

keys = train_df[0].keys()

with open('train_data_splited.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(train_df)

with open('splited_test_data.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(test_df)

# # -----------------------------
dev_df = []
dev_length = len(dev)
for i in range(dev_length):
    curr_data = {}
    curr_data['label'] = dev[i][2]
    sent_1 = '[CLS] ' + dev[i][0] + ' [SEP] '
    sent_2 = dev[i][1] + ' [SEP]'
    sent_1_token = tokenizer.tokenize(sent_1)
    sent_2_token = tokenizer.tokenize(sent_2)
    curr_data['sequence'] = " ".join(sent_1_token + sent_2_token)
    curr_data['attention_mask'] = combine_mask_array(
        [1] * len(sent_1_token + sent_2_token))
    curr_data['token_type'] = combine_mask_array([0] * len(
        sent_1_token) + [1] * len(sent_2_token))
    dev_df.append(curr_data)


keys = dev_df[0].keys()

with open('dev_data.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(dev_df)

# --------------------------------------------------

# start training part


def preprocessing(tok_ids):
    tok_ids = [int(x) for x in tok_ids]
    return tok_ids


TEXT = data.Field(batch_first=True,
                  use_vocab=False,
                  tokenize=split_and_cut,
                  preprocessing=tokenizer.convert_tokens_to_ids,
                  pad_token=pad_token_idx,
                  unk_token=unk_token_idx)

LABEL = data.LabelField()

ATTENTION = data.Field(batch_first=True,
                       use_vocab=False,
                       tokenize=split_and_cut,
                       preprocessing=preprocessing,
                       pad_token=pad_token_idx)

TTYPE = data.Field(batch_first=True,
                   use_vocab=False,
                   tokenize=split_and_cut,
                   preprocessing=preprocessing,
                   pad_token=1)

fields = [('label', LABEL), ('sequence', TEXT),
          ('attention_mask', ATTENTION), ('token_type', TTYPE)]

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='./',
    train='train_data_splited.csv',
    validation='dev_data.csv',
    test='splited_test_data.csv',
    format='csv',
    fields=fields,
    skip_header=True)

train_data_len = len(train_data)

LABEL.build_vocab(train_data)

# -------------------------------

BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.sequence),
    sort_within_batch=False,
    device=device)


bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')


class BERT_BASE_UNCASED_Model(nn.Module):
    def __init__(self,
                 bert_model,
                 output_dim,
                 ):

        super().__init__()

        self.bert = bert_model

        embedding_dim = bert_model.config.to_dict()['hidden_size']

        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, sequence, attn_mask, token_type):

        embedded = self.bert(
            input_ids=sequence, attention_mask=attn_mask, token_type_ids=token_type)[1]

        return self.out(embedded)


OUTPUT_DIM = len(LABEL.vocab)

model = BERT_BASE_UNCASED_Model(bert_model,
                                OUTPUT_DIM,
                                ).to(device)

optimizer = transformers.AdamW(
    model.parameters(), lr=5e-6, eps=1e-8, correct_bias=False)


def get_scheduler(optimizer, warmup_steps):
    scheduler = transformers.get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps)
    return scheduler


criterion = nn.CrossEntropyLoss().to(device)


def accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = (max_preds.squeeze(1) == y).float()
    return correct.sum() / len(y)

# --------------------------------


def train(model, iterator, optimizer, criterion, scheduler):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        sequence = batch.sequence
        attn_mask = batch.attention_mask
        token_type = batch.token_type
        label = batch.label

        predictions = model(sequence, attn_mask, token_type)

        loss = criterion(predictions, label)

        acc = accuracy(predictions, label)

        loss.backward()

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            sequence = batch.sequence
            attn_mask = batch.attention_mask
            token_type = batch.token_type
            labels = batch.label

            predictions = model(sequence, attn_mask, token_type)

            loss = criterion(predictions, labels)

            acc = accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    # print('epoch_acc: ', epoch_acc)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


EPOCHS = 5

warmup_percent = 0.2
total_steps = math.ceil(EPOCHS*train_data_len*1./BATCH_SIZE)
warmup_steps = int(total_steps*warmup_percent)
scheduler = get_scheduler(optimizer, warmup_steps)

for epoch in range(EPOCHS):

    train_loss, train_acc = train(
        model, train_iterator, optimizer, criterion, scheduler)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    print(
        f'\ttrain loss: {train_loss:.3f} | train accuracy: {train_acc*100:.2f}%')
    print(
        f'\t val loss: {valid_loss:.3f} |  val accuracy: {valid_acc*100:.2f}%')

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'test loss: {test_loss:.3f} |  test accuracy: {test_acc*100:.2f}%')

# -----------label dev data-------------
model.eval()
results_dev = []
count = 0
for test_data in dev:
    sent_1 = '[CLS] ' + test_data[0] + ' [SEP] '
    sent_2 = test_data[1] + ' [SEP]'
    sent_1_token = tokenizer.tokenize(sent_1)
    sent_2_token = tokenizer.tokenize(sent_2)
    sents_token = sent_1_token + sent_2_token
    sents_token_to_ids = tokenizer.convert_tokens_to_ids(sents_token)
    combined_token_type = [0] * len(sent_1_token) + [1] * len(sent_2_token)
    attn_mask = [1] * len(sents_token_to_ids)
    sents_token_to_ids = torch.LongTensor(
        sents_token_to_ids).unsqueeze(0).to(device)
    combined_token_type = torch.LongTensor(
        combined_token_type).unsqueeze(0).to(device)
    attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(device)
    predict_label = model(sents_token_to_ids, attn_mask, combined_token_type)
    temp = predict_label.argmax(dim=-1).item()
    predict_label = predict_label.argmax(dim=-1).item()
    predict_label = LABEL.vocab.itos[predict_label]
    results_dev.append(predict_label)
    if int(test_data[2]) == temp:
        count += 1

print(count)
# assert (len(results) == 4850)
results_dev = [int(x) for x in results_dev]
with open('predictions_dev.txt', 'w', encoding='utf-8') as fp:
    for x in results_dev:
        fp.write(str(x) + '\n')


# ----------label test data------------
model.eval()
results = []

for test_data in test:
    sent_1 = '[CLS] ' + test_data[0] + ' [SEP] '
    sent_2 = test_data[1] + ' [SEP]'
    sent_1_token = tokenizer.tokenize(sent_1)
    sent_2_token = tokenizer.tokenize(sent_2)
    sents_token = sent_1_token + sent_2_token
    sents_token_to_ids = tokenizer.convert_tokens_to_ids(sents_token)
    combined_token_type = [0] * len(sent_1_token) + [1] * len(sent_2_token)
    attn_mask = [1] * len(sents_token_to_ids)
    sents_token_to_ids = torch.LongTensor(
        sents_token_to_ids).unsqueeze(0).to(device)
    combined_token_type = torch.LongTensor(
        combined_token_type).unsqueeze(0).to(device)
    attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(device)
    predict_label = model(sents_token_to_ids, attn_mask, combined_token_type)
    predict_label = predict_label.argmax(dim=-1).item()
    predict_label = LABEL.vocab.itos[predict_label]
    results.append(predict_label)

assert (len(results) == 4850)
results = [int(x) for x in results]
with open('upload_predictions.txt', 'w', encoding='utf-8') as fp:
    for x in results:
        fp.write(str(x) + '\n')
