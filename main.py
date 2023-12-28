from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
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

def p_of_positive(dataset):

    import pandas as pd
    df = pd.DataFrame(dataset['train'])

    positive_label = 1
    count_positive = df[df['label'] == positive_label].shape[0]
    total_examples = df.shape[0]
    proportion_positive = count_positive / total_examples

    print(f"Proportion of positive examples: {proportion_positive}")

    return proportion_positive

class AUCModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Additional custom initialization here
        self.a = torch.nn.Parameter(torch.zeros(1))  # Trainable parameter 'a'
        self.b = torch.nn.Parameter(torch.zeros(1)) 
        self.alpha = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        # Load config if not provided
        config = kwargs.pop('config', None)
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

        # Initialize the model
        model = super().from_pretrained(model_name_or_path, *model_args, config=config, **kwargs)

        # Add the custom parameter 'a'
        model.a = torch.nn.Parameter(torch.zeros(1))
        model.b = torch.nn.Parameter(torch.zeros(1)) 
        model.alpha = torch.nn.Parameter(torch.zeros(1))

        return model

    def forward(self, **inputs):
        # Forward pass through the pre-trained model
        outputs = self.model(**inputs)
        return outputs
    
from torch.optim import SGD

class AUCTrainer(Trainer):
    def __init__(self, *args, p=0.5, lambda_reg=1e-3, alpha_lr=1e-3, other_lr=5e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p  # Store the value of 'p'
        self.lambda_reg = lambda_reg  # Regularization constant

        # Initialize the model here if it's not already initialized
        model = kwargs.get("model")
        if model is None:
            raise ValueError("Model not provided to AUCTrainer")

        # Separate the alpha parameter
        alpha_param = [p for p in model.parameters() if p.requires_grad and p is model.alpha]
        other_params = [p for p in model.parameters() if p.requires_grad and p is not model.alpha]

        # Define two separate optimizers
        self.optimizer_alpha = SGD(alpha_param, lr=alpha_lr)
        self.optimizer_others = SGD(other_params, lr=other_lr)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Positive examples
        positive_indices = labels == 1
        positive_logits = logits[positive_indices]
        a_expanded = model.a.expand_as(positive_logits)
        positive_loss = torch.sum((positive_logits - a_expanded) ** 2) * (1 - self.p)

        # Negative examples
        negative_indices = labels == 0
        negative_logits = logits[negative_indices]
        b_expanded = model.b.expand_as(negative_logits)
        negative_loss = torch.sum((negative_logits - b_expanded) ** 2) * self.p

        # Additional loss component adjusted by alpha
        alpha_loss = 2 * ((1 + model.alpha) * self.p * negative_logits.numel() -
                      (1 + model.alpha) * (1 - self.p) * positive_logits.numel())

        # Loss component -p(1-p) * alpha^2
        alpha_squared_loss = -self.p * (1 - self.p) * torch.square(model.alpha)

        # L1 Regularization for all parameters except 'a', 'b', and 'alpha'
        l1_regularization = sum(torch.sum(torch.abs(param)) for name, param in model.named_parameters() if name not in ['a', 'b', 'alpha'])
        l1_regularization *= self.lambda_reg
        
        # Ensure all loss components are at least 1-dimensional
        positive_loss = positive_loss.view(-1)
        negative_loss = negative_loss.view(-1)
        alpha_loss = alpha_loss.view(-1)
        alpha_squared_loss = alpha_squared_loss.view(-1)
        l1_regularization = l1_regularization.view(-1)

        # Total loss calculation
        loss = positive_loss + negative_loss + alpha_loss + alpha_squared_loss + l1_regularization

        # Ensure the total loss is a single scalar value
        loss = loss.sum()

        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass
        outputs = model(**inputs)
        loss = outputs[0]  

        # Backward pass for all parameters except alpha
        self.optimizer_others.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer_others.step()

        # Gradient ascent for alpha
        self.optimizer_alpha.zero_grad()
        (-loss).backward()  # Negative loss for ascent
        self.optimizer_alpha.step()

        return loss.detach()


def main(args,logger):


    model_name_or_path = "gpt2"
    dataset = load_dataset("sst2")
    positive = p_of_positive(dataset)

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

    peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    prompt_tuning_init_text="What is the sentiment of this sentence? \n Positive , Negative.",
    tokenizer_name_or_path=model_name_or_path,
)


    model = AUCModel.from_pretrained(model_name_or_path)
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


    trainer = AUCTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        p=positive
    )

    trainer.train()
    
    


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





