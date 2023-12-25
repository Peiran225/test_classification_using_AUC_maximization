from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit, 
    TaskType,
)
from datasets import load_dataset
import evaluate
import torch
import numpy as np
import argparse
import logging
from transformers import GPT2LMHeadModel,GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss



def main(args,logger):


    model_name_or_path = "gpt2"
    dataset = load_dataset("sst2")

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)


    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        print("pad token id is none")
        tokenizer.pad_token_id = tokenizer.eos_token_id


    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence"], padding=True, truncation=True) #, max_length=None)
        return outputs

    
    tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence"],
            )
    
    

    print("finishing tokeninzing")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    t = tokenized_datasets['train'][0]
    
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    prompt_tuning_init_text="What is the sentiment of this sentence? \n Positive , Negative.",
    tokenizer_name_or_path=model_name_or_path,
)
    


    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

   

    if model_name_or_path == "gpt2":
        model.config.pad_token_id = tokenizer.pad_token_id



   
   # Train 
    training_args = TrainingArguments(
        output_dir="your-name/gpt2-peft-p-tuning",
        learning_rate=1e-3, 
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01, 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        lr_scheduler_type="constant",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    output = trainer.train()
    output.train_loss

    

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=str, default = None)
    parser.add_argument("--log_file", default=None, type=str)
    
    args = parser.parse_args()
    

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(args,logger)





