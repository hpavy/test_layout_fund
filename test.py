from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from datasets import load_dataset



###### Data loading
dataset = load_dataset(
        "nielsr/funsd-layoutlmv3",
        # data_args.dataset_config_name,
        # cache_dir=model_args.cache_dir,
        # token=True if model_args.use_auth_token else None,
    )

##### Tokenizer
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")


##### model loading
model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")


##### Initialiazing the trainer
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

##### training the model

##### 


print('piche')