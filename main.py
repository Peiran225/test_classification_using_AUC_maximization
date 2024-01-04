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


import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

class AUCLOSS(nn.Module):
	def __init__(self, a, b, w, model):
		super(AUCLOSS, self).__init__()
		self.p = 1 / (1 + 0.2)
		self.a = a
		self.b = b
		self.w = w
		self.model = model
	def forward(self, y_pred, y_true):
		'''
		AUC Margin Loss
		'''
		auc_loss = (1 - self.p) * torch.mean((y_pred - self.a)**2 * (1 == y_true).float()) + self.p * torch.mean((y_pred - self.b)**2 * (0 == y_true).float()) + \
		2 * (1+ self.w) * ( torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (1==y_true).float()))) - self.p * (1 - self.p) * self.w**2
		return auc_loss
	def zero_grad(self):
		self.model.zero_grad()
		self.a.grad = None
		self.b.grad = None
		self.w.grad = None
		  
def SGDA(train_set, test_set, model, args):

	Iteration = 0
	p = 1 / (1 + 0.2)
	a = torch.ones(1, requires_grad=True)
	b = torch.zeros(1, requires_grad=True)
	w = torch.zeros(1, requires_grad=True)

	criterion = AUCLOSS(a, b, w, model)
	# criterion = torch.nn.BCELoss()

	model.train()

	for epoch in range(args.epochs):
		for batch in train_set:
			input_ids = batch['input_ids']
			attention_mask = batch['attention_mask']
			labels = batch['labels']
			inputs = 0

			# Pass the batch to your model
			output = model(**batch)
			print(output.logits.size(),)
			output = F.softmax(output.logits,dim=1)
			loss = criterion(output, labels.view(-1,1).float())
			loss.backward()
			print(f'loss is {loss.data}')
			total_norm = 0
			for p in model.parameters():
				if p.grad is not None:
					param_norm = p.grad.data.norm(2)
					total_norm += param_norm.item() ** 2
			total_norm = total_norm ** 0.5
			print(f"Total gradient norm: {total_norm}")

			# Update
			for i, param in enumerate(model.parameters()):
				if param.requires_grad:
					param.data.add_(param.grad.data, alpha= - args.lr)

			model.zero_grad()
			a.data.copy_(a.data - args.lr * a.grad.data)
			b.data.copy_(b.data - args.lr * b.grad.data)
			w.data.copy_(w.data + args.lr2 * w.grad.data)
			w.data  = torch.clamp(w.data, -10, 10)
			criterion.zero_grad()
			print(f'current iteration: {Iteration}\n')
			print(a.data, b.data, w.data)

			if Iteration % args.inLoop == 0:
				model.eval()
				#### testing  #######
				test_pred = []
				test_true = [] 
				with torch.no_grad():
					for batch in test_set:
						input_ids = batch['input_ids']
						attention_mask = batch['attention_mask']
						labels = batch['labels']

						y_pred = model(**batch).logits
						y_pred = F.softmax(y_pred, dim=1)
						test_pred.append(y_pred[:,1].cpu().detach().numpy())
						test_true.append(labels.numpy())

				test_true = np.concatenate(test_true)
				print(len(test_true))
				test_pred = np.concatenate(test_pred)
				print(len(test_true))
				val_auc =  roc_auc_score(test_true, test_pred) 
				print(f'current auc: {val_auc}')
				model.train()

			Iteration += 1
			if Iteration ==50:
				break

def main(args,logger):


	model_name_or_path = "gpt2"
	dataset = load_dataset("sst2")
	
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
	# training_args = TrainingArguments(
	#     output_dir="your-name/gpt2-peft-p-tuning",
	#     learning_rate=1e-3, 
	#     per_device_train_batch_size=32,
	#     per_device_eval_batch_size=32,
	#     num_train_epochs=1,
	#     weight_decay=0.01, 
	#     evaluation_strategy="epoch",
	#     save_strategy="epoch",
	#     load_best_model_at_end=True,
	#     lr_scheduler_type="constant",
	# )


	# trainer = Trainer(
	#     model=model,
	#     args=training_args,
	#     train_dataset=tokenized_datasets["train"],
	#     eval_dataset=tokenized_datasets["validation"],
	#     tokenizer=tokenizer,
	#     data_collator=data_collator,
	#     compute_metrics=compute_metrics
	# )

	# trainer.train()
	
	train_set = tokenized_datasets["train"] 
	test_set = tokenized_datasets["validation"]

	def collate_fn(batch):
		# 'batch' is a list of examples
		input_ids = [item['input_ids'] for item in batch]
		attention_mask = [item['attention_mask'] for item in batch]
		labels = [item['labels'] for item in batch]

		# Convert lists to tensors and pad sequences to the same length
		input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], 
													batch_first=True, padding_value=0)
		attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in attention_mask], 
														batch_first=True, padding_value=0)
		labels = torch.tensor(labels)

		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

	# Create DataLoader with batching
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
	SGDA(train_loader, test_loader, model, args)

	

	


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--similarity", type=str, default = None)
	parser.add_argument("--log_file", default=None, type=str)
	
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
						help='input batch size for training (default: 1000)')
	parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
						help='input batch size for testing (default: 5000)')
	parser.add_argument('--epochs', type=int, default=1, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--worker-size', type=int, default=2, metavar='N',
						help='szie of worker (default: 4)')

	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.1)')
	parser.add_argument('--lr2', type=float, default=0.0001, metavar='LR',
						help='learning rate (default: 0.1)')
	parser.add_argument('--glr', type=float, default=1, metavar='LR',
						help='learning rate (default: 0.1)')
	parser.add_argument('--alpha', type=float, default=0.1, metavar='alpha',
						help='momentum rate alpha')
	parser.add_argument('--beta', type=float, default=0.1, metavar='alpha',
						help='momentum rate beta')
	parser.add_argument('--lmd', type=float, default=0.001, metavar='alpha',
						help='momentum rate beta')
	parser.add_argument('--rho', type=float, default=0.1, metavar='alpha',
						help='momentum rate rho')

	parser.add_argument('--inLoop', type=int, default=1, metavar='S',
						help='inter loop number')
	parser.add_argument('--init', action='store_true', default=False,
						help='For Saving the current Model')
	parser.add_argument('--dataset', type=str, default='mnist',
						help='Dataset for trainig')
	parser.add_argument('--method', type=str, default='fedavg',
						help='Dataset for trainig')
	parser.add_argument('--seed', type=int, default=0, metavar='S',
						help='random seed (default: 1234)')
	parser.add_argument('--port', type=int, default=29505, metavar='S',
						help='random seed (default: 29505)')
	args = parser.parse_args()
	print(args)

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





