import torch
import torch.nn as nn

class AUCLOSS(nn.Module):
	def __init__(self, a, b, w, model,device):
		super(AUCLOSS, self).__init__()
		self.p = 1 / (1 + 0.2)
		# self.p = self.p.to(device)
		self.a = a.to(device)
		self.b = b.to(device)
		self.w = w.to(device)
		self.model = model
	def forward(self, y_pred, y_true):
		'''
		AUC Margin Loss
		'''
		auc_loss = (1 - self.p) * torch.mean((y_pred - self.a)**2 * (1 == y_true).float()) + self.p * torch.mean((y_pred - self.b)**2 * (0 == y_true).float()) + \
		2 * (1+ self.w) * ( torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (1==y_true).float()))) - self.p * (1 - self.p) * self.w**2
		return auc_loss
	# def zero_grad(self):
	# 	self.model.zero_grad()
	# 	self.a.grad = None
	# 	self.b.grad = None
	# 	self.w.grad = None