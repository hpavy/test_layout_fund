from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
import evaluate
from datasets import load_dataset


dataset = load_dataset(
        "nielsr/funsd-layoutlmv3",
        )

tokenizer = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)  # apply_ocr is false because we already have the text inside the boxes
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)
metric = evaluate.load("accuracy")


### Preprocessing
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="yelp_review_classifier",
    eval_strategy="epoch",
    push_to_hub=False,
)


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_test,
        tokenizer=tokenizer,
        )

print("go to train")

trainer.train()