import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import json
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

LABELS = ["other", "header", "question", "answer"]
LABELS2ID = {label: id for id, label in enumerate(LABELS)}
ID2LABELS = {id: label for id, label in enumerate(LABELS)}


# the arguments for the training
lr = 1e-5
weight_decay = 0.01  # to prevent overfitting with the adamW / a bit like an L2 regularisation
num_batchs = 100
num_epochs = 50
batch_size = 4          # maybe we can adjust this one, for now we are not using it
hidden_dropout_prob = 0.  # dropout to avoid overfitting


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
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=4, hidden_dropout_prob=hidden_dropout_prob)

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


class CustomDataset(Dataset):
    def __init__(self, path):
        """
        Args:
            data (list or array-like): Input data.
            labels (list or array-like): Corresponding labels for the data.
        """
        self.path = path
        self.list_files = [name_file[:-5] for name_file in os.listdir(path+"/annotations")]

    def __len__(self):
        """Return the total number of samples."""
        return len(self.list_files)

    def _get_annotations_unencode(self, idx):
        with open(self.path + "/annotations/" + self.list_files[idx] + ".json", 'r') as file:
            data = json.load(file)["form"]
        boxs = []
        tokens = []
        labels = []
        for element in data:
            for mini_box in element["words"]:
                tokens.append(mini_box["text"])
                boxs.append(element["box"])  # we give coordinate of the big box
                labels.append(LABELS2ID[element["label"]])
        return tokens, boxs, labels

    def _get_image(self, idx):
        image = Image.open(self.path + "/images/" + self.list_files[idx] + ".png")
        return image.convert("RGB")

    def __getitem__(self, idx):
        tokens, boxs, labels = self._get_annotations_unencode(idx)
        image = self._get_image(idx)
        return tokenizer(
            image,
            tokens,
            boxes=boxs,
            word_labels=labels,
            return_tensors="pt",
            truncation=True,
            padding=True
            )






# we put everything on the gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# for now we don't use any batch_size
dataset_training = CustomDataset("test_layout_fund/dataset/training_data")
train_dataloader = DataLoader(dataset_training, shuffle=True)

dataset_test = CustomDataset("test_layout_fund/dataset/testing_data")
test_dataloader = DataLoader(dataset_test)

# the training loop
nb_batch = len(train_dataloader)
nb_batch_test = len(test_dataloader)

for epoch in range(num_epochs):
    # test
    model.eval()
    loss_batch_test = 0
    accuracy = 0
    for nb, batch_test in enumerate(test_dataloader):
        encoding = {k: v[0].to(device) for k, v in batch_test.items()} # because we have just one batch
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
        encoding = {k: v[0].to(device) for k, v in batch.items()}  # because we have just one batch
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