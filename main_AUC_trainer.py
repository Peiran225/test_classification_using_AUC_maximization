from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	AutoConfig,
	DataCollatorWithPadding,
	TrainingArguments,
	Trainer,
	TrainerState
)
from transformers.trainer_pt_utils import (
	DistributedTensorGatherer,
	IterableDatasetShard,
	LabelSmoother,
	LengthGroupedSampler,
	SequentialDistributedSampler,
	distributed_broadcast_scalars,
	distributed_concat,
	find_batch_size,
	get_model_param_count,
	get_module_class_from_name,
	get_parameter_names,
	nested_concat,
	nested_detach,
	nested_numpify,
	nested_xla_mesh_reduce,
	reissue_pt_warnings,
)
from transformers.utils import (
	ADAPTER_CONFIG_NAME,
	ADAPTER_SAFE_WEIGHTS_NAME,
	ADAPTER_WEIGHTS_NAME,
	CONFIG_NAME,
	SAFE_WEIGHTS_INDEX_NAME,
	SAFE_WEIGHTS_NAME,
	WEIGHTS_INDEX_NAME,
	WEIGHTS_NAME,
	PushInProgress,
	can_return_loss,
	find_labels,
	is_accelerate_available,
	is_apex_available,
	is_bitsandbytes_available,
	is_datasets_available,
	is_in_notebook,
	is_ipex_available,
	is_peft_available,
	is_safetensors_available,
	is_sagemaker_dp_enabled,
	is_sagemaker_mp_enabled,
	is_torch_compile_available,
	is_torch_neuroncore_available,
	is_torch_tpu_available,
	logging,
	strtobool,
)
from transformers.trainer_utils import (
	PREFIX_CHECKPOINT_DIR,
	BestRun,
	EvalLoopOutput,
	EvalPrediction,
	FSDPOption,
	HPSearchBackend,
	HubStrategy,
	IntervalStrategy,
	PredictionOutput,
	RemoveColumnsCollator,
	ShardedDDPOption,
	TrainerMemoryTracker,
	TrainOutput,
	default_compute_objective,
	denumpify_detensorize,
	enable_full_determinism,
	find_executable_batch_size,
	get_last_checkpoint,
	has_length,
	number_of_arguments,
	seed_worker,
	set_seed,
	speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
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
from packaging import version
from datasets import load_dataset, concatenate_datasets
import evaluate
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import math
import os, json
import importlib.util
import time
from transformers import GPT2LMHeadModel,GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.nn import CrossEntropyLoss
from enum import Enum
from sklearn.metrics import roc_auc_score
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import TrainerCallback
from transformers.optimization import get_scheduler

from utils_AUC import AUCLOSS
import torch.nn.functional as F
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

# class PrintStepCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         print(f"Current step number: {state.global_step}")

if is_accelerate_available():
	from accelerate import Accelerator, skip_first_batches
	from accelerate import __version__ as accelerate_version
	from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

	if version.parse(accelerate_version) > version.parse("0.20.3"):
		from accelerate.utils import (
			load_fsdp_model,
			load_fsdp_optimizer,
			save_fsdp_model,
			save_fsdp_optimizer,
		)

def p_of_positive(dataset):

	import pandas as pd
	df = pd.DataFrame(dataset['train'])

	positive_label = 1
	count_positive = df[df['label'] == positive_label].shape[0]
	total_examples = df.shape[0]
	proportion_positive = count_positive / total_examples

	print(f"Proportion of positive examples: {proportion_positive}")

	return proportion_positive



class AUCTrainer(Trainer):
	def __init__(self, *args, p=1 / (1 + 0.2), lambda_reg=0, learning_rate_1=1e-3, learning_rate_2=1e-3, **kwargs):
		super().__init__(*args, **kwargs)
		self.p = p  # Store the value of 'p'
		self.lambda_reg = lambda_reg  # Regularization constant
		self.a = torch.nn.Parameter(torch.ones(1))# .rand(1))  # Trainable parameter 'a'
		self.b = torch.nn.Parameter(torch.zeros(1)) 
		self.w = torch.nn.Parameter(torch.rand(1))
		self.AUC_optim = "adamw_minimax"
		self.learning_rate_1 = learning_rate_1
		self.learning_rate_2 = learning_rate_2
		


		# Initialize the model here if it's not already initialized
		model = kwargs.get("model")
		if model is None:
			raise ValueError("Model not provided to AUCTrainer")

		
		
	def compute_loss(self, model, inputs, return_outputs=False):
		# criterion = AUCLOSS(self.a, self.b, self.w, model,self.args.device)
		outputs = model(**inputs)
		outputs_softmax = F.softmax(outputs.logits,dim=1)
		y_pred = outputs_softmax[:,1]
		labels = inputs['labels'].view(-1,1).float()
		self.a = self.a.to(self.args.device)
		self.b = self.b.to(self.args.device)
		self.w = self.w.to(self.args.device)

		auc_loss = (1 - self.p) * torch.mean((y_pred - self.a)**2 * (1 == labels).float()) + self.p * torch.mean((y_pred - self.b)**2 * (0 == labels).float()) + \
		2 * (1+ self.w) * ( torch.mean((self.p * y_pred * (0 == labels).float() - (1 - self.p) * y_pred * (1==labels).float()))) - self.p * (1 - self.p) * self.w**2
		
		# loss = criterion(outputs_softmax[:,1], labels.view(-1,1).float())
 
		# compute the regularizer
		L2_norm_square = 0
		for _, p in model.named_parameters():
			if p.requires_grad:
				param_norm = p.data.norm(2)
				L2_norm_square += param_norm.item() ** 2
		# print("params_L2_norm_square %s in compute_loss function "% L2_norm_square) 

		loss = auc_loss +  self.lambda_reg*L2_norm_square
		return (loss, outputs) if return_outputs else loss
	
	
	
	def _inner_training_loop(
		self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
	):
		self.accelerator.free_memory()
		self._train_batch_size = batch_size
		logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
		# Data loader and number of training steps
		train_dataloader = self.get_train_dataloader()

		# Setting up training control variables:
		# number of training epochs: num_train_epochs
		# number of training steps per epoch: num_update_steps_per_epoch
		# total number of training steps to execute: max_steps
		total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

		len_dataloader = None
		if has_length(train_dataloader):
			len_dataloader = len(train_dataloader)
			num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
			num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
			num_examples = self.num_examples(train_dataloader)
			if args.max_steps > 0:
				max_steps = args.max_steps
				num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
					args.max_steps % num_update_steps_per_epoch > 0
				)
				# May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
				# the best we can do.
				num_train_samples = args.max_steps * total_train_batch_size
			else:
				max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
				num_train_epochs = math.ceil(args.num_train_epochs)
				num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
		elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
			max_steps = args.max_steps
			# Setting a very large number of epochs so we go as many times as necessary over the iterator.
			num_train_epochs = sys.maxsize
			num_update_steps_per_epoch = max_steps
			num_examples = total_train_batch_size * args.max_steps
			num_train_samples = args.max_steps * total_train_batch_size
		else:
			raise ValueError(
				"args.max_steps must be set to a positive value if dataloader does not have a length, was"
				f" {args.max_steps}"
			)

		if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
			if self.args.n_gpu > 1:
				# nn.DataParallel(model) replicates the model, creating new variables and module
				# references registered here no longer work on other gpus, breaking the module
				raise ValueError(
					"Currently --debug underflow_overflow is not supported under DP. Please use DDP"
					" (torch.distributed.launch)."
				)
			else:
				debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

		delay_optimizer_creation = (
			self.sharded_ddp is not None
			and self.sharded_ddp != ShardedDDPOption.SIMPLE
			or is_sagemaker_mp_enabled()
			or self.fsdp is not None
			or self.is_fsdp_enabled
		)

		# We need to reset the scheduler, as its parameters may be different on subsequent calls
		if self._created_lr_scheduler:
			self.lr_scheduler = None
			self._created_lr_scheduler = False

		if self.is_deepspeed_enabled:
			self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

		if not delay_optimizer_creation:
			self.create_optimizer_and_scheduler(num_training_steps=max_steps)

		self.state = TrainerState()
		self.state.is_hyper_param_search = trial is not None

		# Compute absolute values for logging, eval, and save if given as ratio
		if args.logging_steps is not None:
			if args.logging_steps < 1:
				self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
			else:
				self.state.logging_steps = args.logging_steps
		if args.eval_steps is not None:
			if args.eval_steps < 1:
				self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
			else:
				self.state.eval_steps = args.eval_steps
		if args.save_steps is not None:
			if args.save_steps < 1:
				self.state.save_steps = math.ceil(max_steps * args.save_steps)
			else:
				self.state.save_steps = args.save_steps

		# Activate gradient checkpointing if needed
		if args.gradient_checkpointing:
			self.model.gradient_checkpointing_enable()

		model = self._wrap_model(self.model_wrapped)

		if (is_sagemaker_mp_enabled() or self.is_fsdp_enabled) and resume_from_checkpoint is not None:
			self._load_from_checkpoint(resume_from_checkpoint, model)

		# as the model is wrapped, don't use `accelerator.prepare`
		# this is for unhandled cases such as
		# Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
		use_accelerator_prepare = True if model is self.model else False

		if delay_optimizer_creation:
			if use_accelerator_prepare:
				self.model = self.accelerator.prepare(self.model)
			self.create_optimizer_and_scheduler(num_training_steps=max_steps)

		# prepare using `accelerator` prepare
		if use_accelerator_prepare:
			self.model.train()
			if hasattr(self.lr_scheduler, "step"):
				if self.use_apex:
					model = self.accelerator.prepare(self.model)
				else:
					model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
			else:
				# to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
				model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
					self.model, self.optimizer, self.lr_scheduler
				)

		if self.is_fsdp_enabled:
			self.model = model

		# for the rest of this function `model` is the outside model, whether it was wrapped or not
		if model is not self.model:
			self.model_wrapped = model

		# backward compatibility
		if self.is_deepspeed_enabled:
			self.deepspeed = self.model_wrapped

		# deepspeed ckpt loading
		if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
			deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

		# Check if saved optimizer or scheduler states exist
		self._load_optimizer_and_scheduler(resume_from_checkpoint)

		# important: at this point:
		# self.model         is the Transformers Model
		# self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

		# Train!
		logger.info("***** Running training *****")
		logger.info(f"  Num examples = {num_examples:,}")
		logger.info(f"  Num Epochs = {num_train_epochs:,}")
		logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
		if self.args.per_device_train_batch_size != self._train_batch_size:
			logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
		logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
		logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
		logger.info(f"  Total optimization steps = {max_steps:,}")
		logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

		self.state.epoch = 0
		start_time = time.time()
		epochs_trained = 0
		steps_trained_in_current_epoch = 0
		steps_trained_progress_bar = None

		# Check if continuing training from a checkpoint
		if resume_from_checkpoint is not None and os.path.isfile(
			os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
		):
			self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
			epochs_trained = self.state.global_step // num_update_steps_per_epoch
			if not args.ignore_data_skip:
				steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
				steps_trained_in_current_epoch *= args.gradient_accumulation_steps
			else:
				steps_trained_in_current_epoch = 0

			logger.info("  Continuing training from checkpoint, will skip to saved global_step")
			logger.info(f"  Continuing training from epoch {epochs_trained}")
			logger.info(f"  Continuing training from global step {self.state.global_step}")
			if not args.ignore_data_skip:
				logger.info(
					f"  Will skip the first {epochs_trained} epochs then the first"
					f" {steps_trained_in_current_epoch} batches in the first epoch."
				)

		# Update the references
		self.callback_handler.model = self.model
		self.callback_handler.optimizer = self.optimizer
		self.callback_handler.lr_scheduler = self.lr_scheduler
		self.callback_handler.train_dataloader = train_dataloader
		if self.hp_name is not None and self._trial is not None:
			# use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
			# parameter to Train when using DDP.
			self.state.trial_name = self.hp_name(self._trial)
		if trial is not None:
			assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
			self.state.trial_params = hp_params(assignments)
		else:
			self.state.trial_params = None
		# This should be the same if the state has been saved but in case the training arguments changed, it's safer
		# to set this after the load.
		self.state.max_steps = max_steps
		self.state.num_train_epochs = num_train_epochs
		self.state.is_local_process_zero = self.is_local_process_zero()
		self.state.is_world_process_zero = self.is_world_process_zero()

		# tr_loss is a tensor to avoid synchronization of TPUs through .item()
		tr_loss = torch.tensor(0.0).to(args.device)
		# _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
		self._total_loss_scalar = 0.0
		self._globalstep_last_logged = self.state.global_step
		model.zero_grad()

		self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

		# Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
		if not args.ignore_data_skip:
			for epoch in range(epochs_trained):
				for _ in train_dataloader:
					break

		total_batched_samples = 0
		for epoch in range(epochs_trained, num_train_epochs):
			epoch_iterator = train_dataloader

			# Reset the past mems state at the beginning of each epoch if necessary.
			if args.past_index >= 0:
				self._past = None

			steps_in_epoch = (
				len(epoch_iterator)
				if len_dataloader is not None
				else args.max_steps * args.gradient_accumulation_steps
			)
			self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

			if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
				self._load_rng_state(resume_from_checkpoint)

			rng_to_sync = False
			steps_skipped = 0
			if steps_trained_in_current_epoch > 0:
				epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
				steps_skipped = steps_trained_in_current_epoch
				steps_trained_in_current_epoch = 0
				rng_to_sync = True

			step = -1
			for step, inputs in enumerate(epoch_iterator):
				total_batched_samples += 1
				if rng_to_sync:
					self._load_rng_state(resume_from_checkpoint)
					rng_to_sync = False

				# Skip past any already trained steps if resuming training
				if steps_trained_in_current_epoch > 0:
					steps_trained_in_current_epoch -= 1
					if steps_trained_progress_bar is not None:
						steps_trained_progress_bar.update(1)
					if steps_trained_in_current_epoch == 0:
						self._load_rng_state(resume_from_checkpoint)
					continue
				elif steps_trained_progress_bar is not None:
					steps_trained_progress_bar.close()
					steps_trained_progress_bar = None

				if step % args.gradient_accumulation_steps == 0:
					self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

				with self.accelerator.accumulate(model):
					tr_loss_step = self.training_step(model, inputs)

				if (
					args.logging_nan_inf_filter
					and not is_torch_tpu_available()
					and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
				):
					# if loss is nan or inf simply add the average of previous logged losses
					tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
				else:
					tr_loss += tr_loss_step

				self.current_flos += float(self.floating_point_ops(inputs))

				is_last_step_and_steps_less_than_grad_acc = (
					steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
				)

				if (
					total_batched_samples % args.gradient_accumulation_steps == 0
					or
					# last step in epoch but step is always smaller than gradient_accumulation_steps
					is_last_step_and_steps_less_than_grad_acc
				):
					# the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
					# in accelerate. So, explicitly enable sync gradients to True in that case.
					if is_last_step_and_steps_less_than_grad_acc or (
						version.parse(accelerate_version) <= version.parse("0.20.3")
					):
						self.accelerator.gradient_state._set_sync_gradients(True)

					# Gradient clipping
					if args.max_grad_norm is not None and args.max_grad_norm > 0:
						# deepspeed does its own clipping

						if self.do_grad_scaling:
							# Reduce gradients first for XLA
							if is_torch_tpu_available():
								gradients = xm._fetch_gradients(self.optimizer)
								xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
							# AMP: gradients need unscaling
							self.scaler.unscale_(self.optimizer)

						if is_sagemaker_mp_enabled() and args.fp16:
							self.optimizer.clip_master_grads(args.max_grad_norm)
						elif hasattr(self.optimizer, "clip_grad_norm"):
							# Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
							self.optimizer.clip_grad_norm(args.max_grad_norm)
						elif hasattr(model, "clip_grad_norm_"):
							# Some models (like FullyShardedDDP) have a specific way to do gradient clipping
							model.clip_grad_norm_(args.max_grad_norm)
						elif self.use_apex:
							# Revert to normal clipping otherwise, handling Apex or full precision
							nn.utils.clip_grad_norm_(
								amp.master_params(self.optimizer),
								args.max_grad_norm,
							)
						else:
							self.accelerator.clip_grad_norm_(
								model.parameters(),
								args.max_grad_norm,
							)

					# Optimizer step
					optimizer_was_run = True
					if is_torch_tpu_available():
						if self.do_grad_scaling:
							self.scaler.step(self.optimizer)
							self.scaler.update()
						else:
							# tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
							self.optimizer.step()
					elif self.do_grad_scaling:
						scale_before = self.scaler.get_scale()
						self.scaler.step(self.optimizer)
						self.scaler.update()
						scale_after = self.scaler.get_scale()
						optimizer_was_run = scale_before <= scale_after
					else:
						# for name, param in model.named_parameters():
						#     if name in ['base_model.a', 'base_model.b', 'base_model.alpha']:
						#         print(f"{name} value:{param.data.item()}, gradient: {param.grad.item()}")
						# Update
						
						total_norm = 0
						# for _, p in model.named_parameters():
						#     if p.requires_grad:
						#         # p.data.add_(p.grad.data, alpha= - self.lr)
						#         a = p
						#         param_norm = p.grad.data.norm(2)
						#         total_norm += param_norm.item() ** 2
								

						# # print("Total params gradient norm square: %s, a norm %s, b norm %s,  w norm %s"% (total_norm,self.a.grad.data.norm(2),self.b.grad.data.norm(2),self.w.grad.data.norm(2)))
						# print("Total params gradient norm square: %s"% total_norm)

						# # Separate the alpha parameter
						# alpha_param = [p for p in model.parameters() if p.requires_grad and p is model.alpha]
						# other_params = [p for p in model.parameters() if p.requires_grad and p is not model.alpha]

						# # Define two separate optimizers
						# self.optimizer_alpha = Adam(alpha_param, lr=alpha_lr, betas=(0.9, 0.999))
						# self.optimizer_others = Adam(other_params, lr=other_lr, betas=(0.9, 0.999))
						
						
						self.optimizer.step() 
						# model.zero_grad()
						# self.a.data.copy_(self.a.data - self.lr * self.a.grad.data)
						# self.b.data.copy_(self.b.data - self.lr * self.b.grad.data)
						# self.w.data.copy_(self.w.data + self.lr2 * self.w.grad.data)
						# self.w.data  = torch.clamp(self.w.data, -10, 10)
						# self.a.data  = torch.clamp(self.w.data, 0, 1)
						# self.b.data  = torch.clamp(self.w.data, 0, 1)
						# self.a.grad.zero_()
						# self.b.grad.zero_()
						# self.w.grad.zero_()
						
						
						# print("a %s b %s w %s"% (self.a.data, self.b.data, self.w.data))

						

						optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

					if optimizer_was_run:
						# Delay optimizer scheduling until metrics are generated
						if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
							self.lr_scheduler.step()

					model.zero_grad()
					self.state.global_step += 1
					self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
					self.control = self.callback_handler.on_step_end(args, self.state, self.control)

					self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
				else:
					self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

				if self.control.should_epoch_stop or self.control.should_training_stop:
					break
			if step < 0:
				logger.warning(
					"There seems to be not a single sample in your epoch_iterator, stopping training at step"
					f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
					f" num_steps ({max_steps}) higher than the number of available samples."
				)
				self.control.should_training_stop = True

			self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
			self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

			if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
				if is_torch_tpu_available():
					# tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
					xm.master_print(met.metrics_report())
				else:
					logger.warning(
						"You enabled PyTorch/XLA debug metrics but you don't have a TPU "
						"configured. Check your training configuration if this is unexpected."
					)
			if self.control.should_training_stop:
				break

		if args.past_index and hasattr(self, "_past"):
			# Clean the state at the end of training
			delattr(self, "_past")

		logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
		if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
			# Wait for everyone to get here so we are sur the model has been saved by process 0.
			if is_torch_tpu_available():
				xm.rendezvous("load_best_model_at_end")
			elif args.parallel_mode == ParallelMode.DISTRIBUTED:
				dist.barrier()
			elif is_sagemaker_mp_enabled():
				smp.barrier()

			self._load_best_model()

		# add remaining tr_loss
		self._total_loss_scalar += tr_loss.item()
		train_loss = self._total_loss_scalar / self.state.global_step

		metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
		self.store_flos()
		metrics["total_flos"] = self.state.total_flos
		metrics["train_loss"] = train_loss

		self.is_in_train = False

		self._memory_tracker.stop_and_update_metrics(metrics)

		self.log(metrics)

		run_dir = self._get_output_dir(trial)
		checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

		# Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
		if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
			for checkpoint in checkpoints_sorted:
				if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
					logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
					shutil.rmtree(checkpoint)

		self.control = self.callback_handler.on_train_end(args, self.state, self.control)

		# Wait for the checkpoint to be uploaded.
		self._finish_current_push()

		return TrainOutput(self.state.global_step, train_loss, metrics)
	
	def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
		"""
		Perform a training step on a batch of inputs.

		Subclass and override to inject custom behavior.

		Args:
			model (`nn.Module`):
				The model to train.
			inputs (`Dict[str, Union[torch.Tensor, Any]]`):
				The inputs and targets of the model.

				The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
				argument `labels`. Check your model's documentation for all accepted arguments.

		Return:
			`torch.Tensor`: The tensor with training loss on this batch.
		"""
		model.train()
		inputs = self._prepare_inputs(inputs)
		
		if is_sagemaker_mp_enabled():
			loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
			return loss_mb.reduce_mean().detach().to(self.args.device)
		
		with self.compute_loss_context_manager():
			loss = self.compute_loss(model, inputs)

		if self.args.n_gpu > 1:
			loss = loss.mean()  # mean() to average on multi-gpu parallel training

		if self.do_grad_scaling:
			self.scaler.scale(loss).backward()
		elif self.use_apex:
			with amp.scale_loss(loss, self.optimizer) as scaled_loss:
				scaled_loss.backward()
		else:
			self.accelerator.backward(loss)
		# total_norm = 0
		
		# for _, p in model.named_parameters():
		#     if p.requires_grad:
		#         param_norm = p.grad.data.norm(2)
		#         total_norm += param_norm.item() ** 2
		# # total_norm = total_norm ** 0.5 + self.a.grad.data.norm(2) + self.b.grad.data.norm(2) + self.w.grad.data.norm(2)
		# print("Total params gradient norm square: %s, a norm %s, b norm %s,  w norm %s"% (total_norm,self.a.grad.data.norm(2),self.b.grad.data.norm(2),self.w.grad.data.norm(2)))

		loss_return =   loss.detach() / self.args.gradient_accumulation_steps
		return torch.tensor(loss_return.item())
	
	def create_scheduler(self, num_training_steps: int, optimizer: None):
		"""
		Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
		passed as an argument.

		Args:
			num_training_steps (int): The number of training steps to do.
		"""
		if self.lr_scheduler is None:
			self.lr_scheduler = get_scheduler(
				self.args.lr_scheduler_type,
				optimizer=self.optimizer if optimizer is None else optimizer,
				num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
				num_training_steps=num_training_steps,
			)
			self._created_lr_scheduler = True
		return self.lr_scheduler
	
	def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
		"""
		Returns the optimizer class and optimizer parameters based on the training arguments.

		Args:
			args (`transformers.training_args.TrainingArguments`):
				The training arguments for the training session.

		"""

		# parse args.optim_args
		optim_args = {}
		if args.optim_args:
			for mapping in args.optim_args.replace(" ", "").split(","):
				key, value = mapping.split("=")
				optim_args[key] = value

		optimizer_kwargs = {"lr1": args.learning_rate1, "lr2": args.learning_rate_2}

		adam_kwargs = {
			"betas": (args.adam_beta1, args.adam_beta2),
			"eps": args.adam_epsilon,
		}
		if args.optim == "adamw_minimax":
			optimizer_kwargs.update(adam_kwargs)
			optimizer_cls = adamw_minimax
		elif args.optim == OptimizerNames.ADAFACTOR:
			optimizer_cls = Adafactor
			optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
		elif args.optim == OptimizerNames.ADAMW_HF:
			from .optimization import AdamW

			optimizer_cls = AdamW
			optimizer_kwargs.update(adam_kwargs)
		elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
			from torch.optim import AdamW

			optimizer_cls = AdamW
			optimizer_kwargs.update(adam_kwargs)
			if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
				optimizer_kwargs.update({"fused": True})
		elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
			try:
				from torch_xla.amp.syncfree import AdamW

				optimizer_cls = AdamW
				optimizer_kwargs.update(adam_kwargs)
			except ImportError:
				raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
		elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
			try:
				from apex.optimizers import FusedAdam

				optimizer_cls = FusedAdam
				optimizer_kwargs.update(adam_kwargs)
			except ImportError:
				raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
		elif args.optim in [
			OptimizerNames.ADAMW_BNB,
			OptimizerNames.ADAMW_8BIT,
			OptimizerNames.PAGED_ADAMW,
			OptimizerNames.PAGED_ADAMW_8BIT,
			OptimizerNames.LION,
			OptimizerNames.LION_8BIT,
			OptimizerNames.PAGED_LION,
			OptimizerNames.PAGED_LION_8BIT,
		]:
			try:
				from bitsandbytes.optim import AdamW, Lion

				is_paged = False
				optim_bits = 32
				optimizer_cls = None
				additional_optim_kwargs = adam_kwargs
				if "paged" in args.optim:
					is_paged = True
				if "8bit" in args.optim:
					optim_bits = 8
				if "adam" in args.optim:
					optimizer_cls = AdamW
				elif "lion" in args.optim:
					optimizer_cls = Lion
					additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}

				bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
				optimizer_kwargs.update(additional_optim_kwargs)
				optimizer_kwargs.update(bnb_kwargs)
			except ImportError:
				raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
			if is_bitsandbytes_available() and version.parse(
				importlib.metadata.version("bitsandbytes")
			) < version.parse("0.41.1"):
				logger.warning(
					"You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
					"It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
				)
		elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
			try:
				from torchdistx.optimizers import AnyPrecisionAdamW

				optimizer_cls = AnyPrecisionAdamW
				optimizer_kwargs.update(adam_kwargs)

				# TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
				optimizer_kwargs.update(
					{
						"use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
						"momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
						"variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
						"compensation_buffer_dtype": getattr(
							torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
						),
					}
				)
			except ImportError:
				raise ValueError("Please install https://github.com/pytorch/torchdistx")
		elif args.optim == OptimizerNames.SGD:
			optimizer_cls = torch.optim.SGD
		elif args.optim == OptimizerNames.ADAGRAD:
			optimizer_cls = torch.optim.Adagrad
		else:
			raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
		return optimizer_cls, optimizer_kwargs

	def create_optimizer(self):
		"""
		Setup the optimizer.

		We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
		Trainer's init through `optimizers`, or subclass and override this method in a subclass.
		"""
		opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

		if self.optimizer is None:
			decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
			decay_parameters = [name for name in decay_parameters if "bias" not in name]
			optimizer_grouped_parameters = [
				{
					"params": [
						p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
					],
					"weight_decay": self.args.weight_decay,
					"lr": self.learning_rate_1,
					"betas": (self.args.adam_beta1, self.args.adam_beta2),
					"eps": self.args.adam_epsilon,
				},
				{
					"params": [
						p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
					],
					"weight_decay": 0.0,
					"lr": self.learning_rate_1,
					"betas": (self.args.adam_beta1, self.args.adam_beta2),
					"eps": self.args.adam_epsilon,
				},
				{
					"params": [
						self.a, self.b
					],
					"weight_decay": 0.0,
					"lr": self.learning_rate_1,
					"betas": (self.args.adam_beta1, self.args.adam_beta2),
					"eps": self.args.adam_epsilon,
				},
				{
					"params": [
						self.w
					],
					"weight_decay": 0.0,
					"lr":  - self.learning_rate_2,
					"betas": (self.args.adam_beta1, self.args.adam_beta2),
					"eps": self.args.adam_epsilon,
				},
			]

			optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

			if self.sharded_ddp == ShardedDDPOption.SIMPLE:
				self.optimizer = OSS(
					params=optimizer_grouped_parameters,
					optim=optimizer_cls,
					**optimizer_kwargs,
				)
			elif self.AUC_optim == "adamw_minimax":
				self.optimizer = optimizer_cls(optimizer_grouped_parameters)
			else:
				self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
				if optimizer_cls.__name__ == "Adam8bit":
					import bitsandbytes

					manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

					skipped = 0
					for module in opt_model.modules():
						if isinstance(module, nn.Embedding):
							skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
							logger.info(f"skipped {module}: {skipped/2**20}M params")
							manager.register_module_override(module, "weight", {"optim_bits": 32})
							logger.debug(f"bitsandbytes: will optimize {module} in fp32")
					logger.info(f"skipped: {skipped/2**20}M params")

		if is_sagemaker_mp_enabled():
			self.optimizer = smp.DistributedOptimizer(self.optimizer)

		return self.optimizer

		
		





def main(args):


	model_name_or_path = "gpt2"
	dataset = load_dataset("sst2")

	positive_samples = dataset['train'].filter(lambda example: example['label'] == 1)
	negative_samples = dataset['train'].filter(lambda example: example['label'] == 0)
	num_positive_to_keep = int(0.16 * len(positive_samples))
	reduced_positive_samples = positive_samples.shuffle(seed=42).select(range(num_positive_to_keep))
	dataset['train'] = concatenate_datasets([negative_samples, reduced_positive_samples]).shuffle(seed=42)
	positive = p_of_positive(dataset)

	metric = evaluate.load('roc_auc') #("accuracy")
	def compute_metrics(eval_pred):
		predictions, labels = eval_pred
		predictions_tensor = torch.tensor(predictions)
		predictions_tensor = F.softmax(predictions_tensor,dim=1)
		predictions_numpy = predictions_tensor.numpy()
		prediction_scores = np.array(predictions_numpy, dtype='float32')

		
		return metric.compute(prediction_scores=prediction_scores[:,1], references=labels)

	


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
	
	ini_prompt = "classify the sentiment type of this"
	org_input = tokenizer(ini_prompt
						  , return_tensors='pt')
	num_virtual_tokens = len(org_input['input_ids'][0])

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

	peft_config = PromptTuningConfig(
	task_type=TaskType.SEQ_CLS,
	prompt_tuning_init=PromptTuningInit.RANDOM, #.TEXT
	num_virtual_tokens=args.num_virtual_tokens,
	# prompt_tuning_init_text=ini_prompt,
	tokenizer_name_or_path=model_name_or_path,
)


	model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
	model = get_peft_model(model, peft_config)
	model.print_trainable_parameters()

   

	if model_name_or_path == "gpt2":
		model.config.pad_token_id = tokenizer.pad_token_id



   
   # Train 
	

	training_args = TrainingArguments(
		output_dir="your-name/gpt2-peft-prompt-tuning",
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		num_train_epochs=args.num_train_epochs,
		weight_decay=args.weight_decay, #originally 0.01
		evaluation_strategy="steps",
		save_strategy="steps",
		save_steps=10,
		load_best_model_at_end=True,
		lr_scheduler_type = args.scheduler_type,
		seed=42,
		data_seed=42,
	)
	print(f'current lr1 {args.learning_rate_1} lr2 {args.learning_rate_2} weight decay {args.weight_decay}')

	trainer = AUCTrainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_datasets["train"],
		eval_dataset=tokenized_datasets["validation"],
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		p=positive,
		learning_rate_1 = args.learning_rate_1,
		learning_rate_2 = args.learning_rate_2,
		# callbacks=[PrintStepCallback()]
	)
	# L2_norm_square = 0
	# L2_norm_sq_grad = 0
	# for _, p in model.named_parameters():
	#         # p0 = p
	#         if p.requires_grad:
	#             # a = p.grad
	#             param_norm = p.data.norm(2)
	#             # param_norm_grad = p.grad.data.norm(2)
	#             L2_norm_square += param_norm.item() ** 2
	#             # L2_norm_square_sq_grad += param_norm_grad.item() ** 2
	#             break
	# print("L2_norm_square %s before train"% L2_norm_square) 
	# print("L2_norm_sq_grad %s before train"% L2_norm_sq_grad)
	eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
	print("balanced data AUC before train\n %s"% eval)

	trainer.train()
	eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
	print("balanced data AUC after train\n %s"% eval)
	
	


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--log_file", default=None, type=str)
	parser.add_argument("--num_virtual_tokens", default=5, type=int)
	parser.add_argument("--learning_rate_1", default=1e-3, type=float)
	parser.add_argument("--learning_rate_2", default=1e-3, type=float)
	parser.add_argument("--weight_decay", default=1e-2, type=float)
	parser.add_argument("--per_device_train_batch_size", default=32, type=int)
	parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
	parser.add_argument("--num_train_epochs", default=5, type=int)
	parser.add_argument("--warmup_ratio", default=0.1, type=float)
	parser.add_argument("--scheduler_type", default='linear', type=str)

	args = parser.parse_args()
	args.warmup_ratio = 0.1
	

	handlers = [logging.StreamHandler()]
	if args.log_file is not None:
		handlers.append(logging.FileHandler(args.log_file))
	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO,
						handlers=handlers)
	logger = logging.getLogger(__name__)
	logger.info(args)
	# print(args)
	
	s_t = ['constant']
	lr1 = [1e-4]
	lr2 = [0.1]
	wd = [1e-1]
	args.num_train_epochs = 5
	for args.learning_rate_1 in lr1:
		for args.learning_rate_2 in lr2:
			for args.weight_decay in wd: 
				for args.scheduler_type in s_t:
					print(f'scheduler type {args.scheduler_type}')
					# if args.learning_rate_1 == 0.001:
					# 	args.num_train_epochs = 20
					main(args)




