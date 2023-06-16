import torch
from datasets import load_dataset, Dataset
import sys
import re
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

model_path = "./word_lm.pt"

# tokenizer arguments
max_seq_length = 32
unk_token = "[unknown_word]"
pad_token = "[padding]"

# training arguments
num_epochs = 50
batch_size = 32
learning_rate = 0.01

# model arguments
emb_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


class MyLM(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, emb_dim)
        self.backbone = nn.LSTM(emb_dim, emb_dim, 3)
        self.vocab_size = vocab_size

    def infer(self, input_ids):
        preds = self.forward({"input_ids": input_ids})
        pred_ids = torch.argmax(preds, dim=-1)
        return pred_ids

    def forward(self, batch):
        # batch["input_ids"] has shape of (batch_size x max_seq_len)
        # embeds has shape of (batch_size x max_seq_len x emb_dim)
        # projections has shape of (batch_size x max_seq_len x emb_dim)
        # embeddings.weight has shape of (vocab_size x emb_dim)
        # pred has shape of (batch_size x max_seq_len x vocab_size)
        embeds = self.embeddings(batch["input_ids"])
        projections, _ = self.backbone(embeds)
        batch_size = projections.shape[0]
        emb_dim = projections.shape[-1]
        inverse_embeddings = self.embeddings.weight.T.unsqueeze(
            0).expand(batch_size, emb_dim, self.vocab_size)
        preds = torch.bmm(projections, inverse_embeddings)
        return preds

    def get_loss(self, preds, batch):
        vocab_size = preds.shape[-1]

        actual_preds = preds[:, :-1, :]
        actual_labels = batch["input_ids"][:, 1:]

        # masks has shape of (batch_size x max_seq_len)
        masks = batch["masks"][:, 1:].unsqueeze(-1).expand(actual_preds.shape)
        # masks has now shape of (batch_size x (max_seq_len - 1) x vocab_size)

        actual_pred_probs = F.log_softmax(actual_preds, dim=-1) * masks
        actual_pred_probs, actual_labels = actual_pred_probs.reshape(
            -1, vocab_size), actual_labels.reshape(-1)
        loss = torch.nn.NLLLoss()(actual_pred_probs, actual_labels)
        return loss


class Dataset():
    def __init__(self, vocab_file=None) -> None:
        dataset = load_dataset("tiny_shakespeare")

        if vocab_file is None:
            self.create_vocabulary(dataset)
            vocab_file = "./vocab.txt"
        with open(vocab_file, "r") as vocab_file:
            vocabulary = [line.strip() for line in vocab_file.readlines()]
            self.id_to_vocab = {idx: vocab for idx,
                                vocab in enumerate(vocabulary)}
            self.vocab_to_id = {vocab: idx for idx,
                                vocab in enumerate(vocabulary)}
            self.vocab_size = len(vocabulary)

        self.lm_dataset = self.create_lm_dataset(dataset)

    def create_vocabulary(self, dataset):
        # clean initial text
        all_text = self.clean_text(dataset["train"]["text"][0])
        all_words = all_text.split(" ")
        all_words_lower = [word for word in all_words]

        # create vocabulary
        vocabulary = set(all_words_lower)
        vocabulary.add(unk_token)
        vocabulary.add(pad_token)
        vocabulary = list(vocabulary)

        # dictionary utils for converting word to ids and ids to words
        self.id_to_vocab = {idx: vocab for idx, vocab in enumerate(vocabulary)}
        self.vocab_to_id = {vocab: idx for idx, vocab in enumerate(vocabulary)}

        # save vocabulary
        with open("vocab.txt", "w") as vocab_file:
            for vocab in vocabulary:
                vocab_file.write(vocab)
                vocab_file.write("\n")

    def clean_text(self, text):
        text = text.lower()
        text = text.replace("\n", " ")
        text = re.sub(r" +", " ", text)
        return text

    def tokenize(self, text):
        text = self.clean_text(text)
        words = text.split(" ")
        word_ids = [self.vocab_to_id.get(
            word, self.vocab_to_id[unk_token]) for word in words]
        if len(word_ids) > max_seq_length:
            word_ids = word_ids[:max_seq_length]
        elif len(word_ids) < max_seq_length:
            word_ids = word_ids + \
                [self.vocab_to_id[pad_token]] * \
                (max_seq_length - len(word_ids))
        return word_ids

    def create_lm_dataset(self, dataset):
        # begin section : converting original dataset into language modelling dataset

        # refer : https://huggingface.co/docs/datasets/process#split-long-examples
        def create_lm_dataset(examples):
            words = examples["text"][0].split(" ")
            training_examples = [
                " ".join(words[start_idx:start_idx + max_seq_length])
                for start_idx in range(0, len(words), max_seq_length)
            ]
            tokenized_input_ids = torch.Tensor(
                [self.tokenize(text) for text in training_examples]).long()
            masks = [input_ids != self.vocab_to_id[pad_token]
                     for input_ids in tokenized_input_ids]
            return {
                "input_ids": tokenized_input_ids,
                "masks": masks,
                "input_texts": training_examples
            }
        lm_dataset = dataset.map(
            create_lm_dataset, batched=True, remove_columns=["text"])
        lm_dataset.set_format(type="torch", columns=["input_ids", "masks"])
        # end section

        return lm_dataset

    def get_dataloader(self, mode):
        shuffle = True if mode == "train" else False
        return DataLoader(self.lm_dataset[mode], batch_size=batch_size, shuffle=shuffle)


def train():
    dataset = Dataset()
    model = MyLM(dataset.vocab_size)

    print(f"Starting training using device : {device}")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    num_steps = 0
    train_loss = []
    evalLoss = []
    model = model.to(device)
    print(model)
    for i in range(num_epochs):
        count = 0
        tloss = 0
        for batch in dataset.get_dataloader("train"):
            count += 1
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["masks"] = batch["masks"].to(device)
            model.train()
            optimizer.zero_grad()
            preds = model(batch)
            loss = model.get_loss(preds, batch)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            num_steps += 1

            if num_steps % 100 == 0:
                model.eval()
                with torch.no_grad():
                    eval_loss = []
                    for batch in dataset.get_dataloader("test"):
                        batch["input_ids"] = batch["input_ids"].to(device)
                        batch["masks"] = batch["masks"].to(device)
                        preds = model(batch)
                        loss = model.get_loss(preds, batch)
                        eval_loss.append(loss.item())
                    print(
                        f"Train loss : {loss}, Eval loss at iteration {i}, steps {num_steps} : {np.mean(eval_loss)}")
        train_loss.append(tloss/count)
        eval_loss = []
        for batch in dataset.get_dataloader("test"):
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["masks"] = batch["masks"].to(device)
            preds = model(batch)
            loss = model.get_loss(preds, batch)
            eval_loss.append(loss.item())
        evalLoss.append(np.mean(eval_loss))
    torch.save(model, model_path)
    plt.figure()
    # print(train_loss,type(train_loss))
    plt.plot(train_loss, label="training")
    plt.plot(evalLoss, label="evaluation")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("Training Loss")
    plt.savefig("WordModelLoss.jpg")


def infer():
    input_text = "take the priest, clerk, and some sufficient honest"
    model = torch.load(model_path)
    dataset = Dataset(vocab_file="./vocab.txt")

    input_ids = dataset.tokenize(input_text)
    len_input = input_ids.index(dataset.vocab_to_id[pad_token])
    print(" ".join([dataset.id_to_vocab[word_id] for word_id in input_ids]))
    input_ids = torch.Tensor([input_ids]).long().to(device)

    preds = model({"input_ids": input_ids})
    pred_ids = torch.argmax(preds[0], dim=-1)
    predicted_text = " ".join([dataset.id_to_vocab[pred_id.item()]
                              for pred_id in pred_ids[len_input:]])
    print(predicted_text)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        train()
    elif mode == "infer":
        infer()
