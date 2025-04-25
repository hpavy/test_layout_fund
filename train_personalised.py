from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments
)
import torch
import torch.optim as optim
from datasets import load_dataset


# charge the dataset
dataset = load_dataset(
        "nielsr/funsd-layoutlmv3",
        )


# the arguments for the training
lr = 1e-5
weight_decay = 0.01  # to prevent overfitting with the adamW / a bit like an L2 regularisation
num_batchs = 100
num_epochs = 50
batch_size = 4          # maybe we can adjust this one, for now we are not using it
hidden_dropout_prob = 0.5  # dropout to avoid overfitting


# compute the loss manually with cross entropy
# we don't use it, just to understand the loss and check it is the same
def loss_cross_entropy(return_model, encoding):
    masque_special_item = ~(encoding["labels"]==-100)  # to remove special tokens
    proba = return_model.logits[masque_special_item].softmax(dim=-1)
    labels_find = encoding["labels"][masque_special_item]
    result = -torch.log(proba[torch.arange(labels_find.shape[0]), labels_find])
    return result.mean()


def compute_accuracy(result, encoding):
    """ return the percentage of right predictions """
    masque_special_item = ~(encoding["labels"]==-100)  # to remove special tokens
    labels_predict = torch.argmax(result[masque_special_item], dim=-1)
    labels_true = encoding['labels'][masque_special_item]
    return ((labels_predict == labels_true).sum()/labels_predict.shape[0]).item()


# charge the tokenizer and the model
tokenizer = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)  # apply_ocr is false because we already have the text inside the boxes
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7, hidden_dropout_prob=hidden_dropout_prob)


# charge an optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# Preprocessing for the dataset
def tokenize_function(example):
    return tokenizer(
        example["image"],
        example['tokens'],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        return_tensors="pt",
        truncation=True,
        padding=True
        )


tokenized_datasets_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_datasets_test = dataset["test"].map(tokenize_function, batched=True)

# we put everything on the gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# for now we don't use any batch_size
train_dataloader = tokenized_datasets_train
test_dataloader = tokenized_datasets_test

# the training loop
nb_batch = len(train_dataloader)
nb_batch_test = len(test_dataloader)

for epoch in range(num_epochs):
    # test
    model.eval()
    loss_batch_test = 0
    accuracy = 0
    for nb, batch_test in enumerate(test_dataloader):
        tok = batch_test['tokens']
        image = batch_test["image"]
        labels = batch_test["ner_tags"]
        box = batch_test["bboxes"]
        encoding = tokenizer(image, tok, boxes=box, word_labels=labels, return_tensors="pt", truncation=True, padding=False)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        return_model = model(**encoding)
        loss, result = return_model.values()
        loss_batch_test += loss.item()
        accuracy_batch = compute_accuracy(result, encoding)
        accuracy = (accuracy*nb + accuracy_batch) / (nb + 1)

    loss_epoch_test = loss_batch_test / nb_batch_test
    print(f"Test:  Epoch: {epoch}, Loss: {loss_epoch_test:.1e}, Accuracy: {accuracy:.2f}")

    # train
    model.train()  # Set the model to training mode
    loss_batch = 0
    ############
    # accuracy = 0
    ########
    for batch in train_dataloader:
        tok = batch['tokens']
        image = batch["image"]
        labels = batch["ner_tags"]
        box = batch["bboxes"]
        encoding = tokenizer(image, tok, boxes=box, word_labels=labels, return_tensors="pt", truncation=True, padding=False)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        return_model = model(**encoding)
        loss, result = return_model.values()
        loss_batch += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #########
        # accuracy_batch = compute_accuracy(result, encoding)
        # accuracy = (accuracy*nb + accuracy_batch) / (nb + 1)
        ##########
    loss_epoch = loss_batch / nb_batch
    print(f"Train: Epoch: {epoch}, Loss: {loss_epoch:.1e}")
    # print(f"accuracy:{accuracy:.2f}")

print("Training complete!")
