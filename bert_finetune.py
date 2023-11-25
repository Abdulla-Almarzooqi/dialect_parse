import pandas as pd
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch
import random
import shutil
import math
import einops
from torch.utils.data import random_split
from torch.optim import AdamW
import numpy as np
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count(), flush=True)

    print('We will use the GPU:', torch.cuda.get_device_name(0), flush=True)

# If not...
else:
    print('No GPU available, using the CPU instead.', flush=True)
    device = torch.device("cpu")

def save_checkpoint(model, tokenizer, is_best, model_name,filename='checkpoint'):
    model.save_pretrained(model_name+"_"+filename)
    tokenizer.save_pretrained(model_name+"_"+filename)
    if is_best:
        shutil.copytree(model_name+"_"+filename, model_name+"_"+'best_model', dirs_exist_ok=True)

def get_word_level_embeddings(tokenizer, model, input_text):
    if type(input_text) is tuple:
        input_text = [text.split(' ') for text in input_text]
    else:
        input_text = input_text.split(' ')
    # Tokenize the input text
    tokens = tokenizer(input_text, return_tensors="pt", padding=True, is_split_into_words=True).to(device)
    # Forward pass through the model
    outputs = model(**tokens)
    # Get the last layer hidden states
    hidden_states = outputs[2]
    # Extract the wordpiece-level embeddings by summing last 4 hidden layers
    wordpiece_embeddings = torch.stack(hidden_states[-4:]).sum(0).to(device)

    # Get the attention mask
    attention_mask = tokens["attention_mask"]

    # Get the word IDs
    # word_ids = [token.word_ids() for token in tokens]
    word_ids = [tokens[i].word_ids for i in range(len(tokens["input_ids"]))]
    word_ids = [[ID if ID is not None else len(sent) for ID in sent] for sent in word_ids]
    word_ids = torch.tensor(word_ids).to(device)
    num_words = [len(set(sent.tolist()))-1 for sent in word_ids]

    ranges = torch.stack([torch.arange(max(num_words)) for i in range(len(word_ids))]).to(device)
    # Calculate mean-pooled vectors for each original word using matrix operations
    mask = (einops.repeat(word_ids, 'm n -> m k n', k=max(num_words)) == einops.repeat(ranges, 'm n -> m n k', k=wordpiece_embeddings.shape[1])).float().to(device)
    # Use word IDs to accumulate the embeddings for each word
    word_level_embeddings = torch.bmm(mask, wordpiece_embeddings)

    # Normalize by the number of wordpieces in each word
    word_level_embeddings /= attention_mask.sum()

    return word_level_embeddings

class AlignedTokensDataset(Dataset):
    def __init__(self, csv_file, bert_model="CAMeL-Lab/bert-base-arabic-camelbert-mix", max_sent_length=150):
        df = pd.read_csv(csv_file,header=None,sep='\t')
        df.columns = ['reference_token_id', 'reference_sent', 'aligning_token_id', 'aligning_sent']
        df = df[df.apply((lambda x : len(x['reference_sent'].split(' '))<=max_sent_length), axis=1)]
        df = df[df.apply((lambda x : len(x['aligning_sent'].split(' '))<=max_sent_length), axis=1)]
        self.data = df
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model, output_hidden_states = True).requires_grad_(False).cuda()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        reference_token = self.data['reference_token_id'].iloc[idx]

        reference_sent = self.data['reference_sent'].iloc[idx]
        reference_embedding = 0
        with torch.no_grad():
            if type(idx) is list:
                embeddings = get_word_level_embeddings(self.tokenizer, self.model, reference_sent)
                reference_embedding = torch.stack([embeddings[i,ref,:] for i,ref in enumerate(reference_token-1)])
            else:
                embeddings = get_word_level_embeddings(self.tokenizer, self.model, reference_sent)
#                 print(reference_token, len(reference_sent.split(' ')), embeddings.shape)
                reference_embedding = embeddings[0,reference_token-1,:]
        aligning_token = self.data['aligning_token_id'].iloc[idx]-1
        aligning_sent = self.data['aligning_sent'].iloc[idx]

        return aligning_token, aligning_sent, reference_embedding


NUM_EPOCHS=4
BATCH_SIZE = 64
BERT_MODEL="CAMeL-Lab/bert-base-arabic-camelbert-mix"
MODEL_NAME = "gumar_aligned_msa"

best_val1 = math.inf

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


finetune_data = AlignedTokensDataset("gumar_aligned_msa_paired_tokens.tsv",BERT_MODEL)
train_size = int(0.9 * len(finetune_data))
val_size = len(finetune_data) - train_size
train_dataset, val_dataset = random_split(finetune_data, [train_size, val_size])


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

model = AutoModel.from_pretrained(BERT_MODEL, output_hidden_states = True)
model.cuda()


optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


total_steps = len(train_dataloader) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
loss = MSELoss()



for epoch_i in range(0, NUM_EPOCHS):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, NUM_EPOCHS))
    print('Training...', flush=True)

    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):


        if step % 5 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)), flush=True)
        batch_aligning_toks = batch[0].to(device)
        batch_aligning_sents = batch[1]
        batch_reference_embeddings = batch[2].to(device)
        model.zero_grad()
        output_embeddings = get_word_level_embeddings(tokenizer, model, batch_aligning_sents).to(device)
        token_embeddings = torch.stack([output_embeddings[i,ref,:] for i,ref in enumerate(batch_aligning_toks-1)]).to(device)

        loss_out = loss(token_embeddings, batch_reference_embeddings)
        total_train_loss += loss_out.item()
        loss_out.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("")
    print("Running Validation...", flush=True)
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    for batch in validation_dataloader:
        batch_aligning_toks = batch[0].to(device)
        batch_aligning_sents = batch[1]
        batch_reference_embeddings = batch[2].to(device)
        with torch.no_grad():
            output_embeddings = get_word_level_embeddings(tokenizer, model, batch_aligning_sents).to(device)
            token_embeddings = torch.stack([output_embeddings[i,ref,:] for i,ref in enumerate(batch_aligning_toks-1)]).to(device)
        loss_out = loss(token_embeddings, batch_reference_embeddings)
        total_eval_loss += loss_out.item()
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    is_best = avg_val_loss < best_val1
    best_val1 = min(avg_val_loss, best_val1)
    save_checkpoint(model,tokenizer, is_best, MODEL_NAME)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
print("")
print("Training complete!", flush=True)

