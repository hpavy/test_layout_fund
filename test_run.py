from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from datasets import load_dataset



###### Data loading
dataset = load_dataset(
        "nielsr/funsd-layoutlmv3",
        # data_args.dataset_config_name,
        # cache_dir=model_args.cache_dir,
        # token=True if model_args.use_auth_token else None,
    )

##### Tokenizer
# processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")

tokenizer = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False) # apply_ocr is false because we already have the text inside the boxes

#####
## Maybe we will need to use the ocr classification for our application
#####


##### model loading
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)


#### Load an example
example = dataset["train"][0]
tok = example['tokens']
image = example["image"]
labels = example["ner_tags"]
box = example["bboxes"]
encoding = tokenizer(image, tok, boxes=box, word_labels=labels, return_tensors="pt", truncation=True, padding=False)
return_model = model(**encoding)

#### we predict the label
logits = return_model.logits
proba = logits.softmax(dim=-1)
indices_label = proba.max(-1).indices.squeeze()

masque_special_item = ~(encoding["labels"]==-100).squeeze()
predictions = indices_label[masque_special_item]   # we have the predicted labels 


print('piche')