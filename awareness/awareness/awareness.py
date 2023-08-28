import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Awareness(nn.Module):

	def __init__(self, R=1, learnable=True, dynamic_ray=True):
		super().__init__()

		self.R = R
		self.dynamic_ray = dynamic_ray
		self.awareness = AwarenessLayer(R=R, learnable=learnable, dynamic_ray=dynamic_ray).to(device)

		pass

	# Input format: (B,N,C,W,H)
	def forward(self, x, set_labels=False, update_ref_insts=False):

		batch_labels_list = []

		if(set_labels is False):
			labels = [None]*x.size(0)
		else:
			labels = set_labels

		for (x_batch_item, label) in zip(x, labels):
			pred = self.awareness(x_batch_item, label=label, update_ref_insts=update_ref_insts)
			batch_labels_list.append(pred)

		x_batch_labels_list = torch.Tensor(batch_labels_list)
		
		return x_batch_labels_list

class AwarenessLayer(nn.Module):

	def __init__(self, R=1, learnable=True, dynamic_ray=True):
		super().__init__()

		self.R = R
		self.dynamic_ray = dynamic_ray

		self.ref_insts = None
		self.ref_insts_ids = []
		self.ref_insts_labels = []
		self.ref_insts_freqs = []

		self.curr_id = None
		self.curr_label = None

		self.min_dist = +float('inf')
		self.max_dist = 0

		self.count = 1

		pass

	def updateDistanceBounds(self, dist_mat):

		tmp_min_dist = torch.min(dist_mat)
		tmp_max_dist = torch.max(dist_mat)

		if(tmp_min_dist < self.min_dist):
			self.min_dist = tmp_min_dist

		if(tmp_max_dist > self.max_dist):
			self.max_dist = tmp_max_dist

		self.R = (self.min_dist+self.max_dist)/3

		pass

	def updateReferenceInstances(self, x, label, update_ref_insts):

		if(self.count==1):
			if(update_ref_insts):
				self.ref_insts= x
				self.ref_insts_ids.append(self.count)
				self.ref_insts_labels.append(label)
				self.curr_id = self.ref_insts_ids[-1]
				self.curr_label = self.ref_insts_labels[-1]
		else:
			candidate = x

			dist_mat = torch.cdist(self.ref_insts, candidate)
			dist_mat_logic = dist_mat > self.R

			new_reference = not False in dist_mat_logic

			if(new_reference):
				if(update_ref_insts):
					self.ref_insts = torch.cat((self.ref_insts, candidate))
					self.ref_insts_ids.append(self.count)
					self.ref_insts_labels.append(label)
					self.curr_id = self.ref_insts_ids[-1]
					self.curr_label = self.ref_insts_labels[-1]
			else:
				index = torch.argmin(dist_mat_logic.long())
				self.curr_id = self.ref_insts_ids[index]
				self.curr_label = self.ref_insts_labels[index]

			if(self.dynamic_ray):
				self.updateDistanceBounds(dist_mat)

		pass

	def forward(self, x, label=None, update_ref_insts=False):

		if((len(self.ref_insts_labels) == 0) and (update_ref_insts is False)):
			return None
		else:
			if(update_ref_insts is True):
				self.updateReferenceInstances(x, label, update_ref_insts)

			ref_inst_index = torch.argmin(torch.cdist(x, self.ref_insts))
			pred = self.ref_insts_labels[ref_inst_index].item()

			self.count = self.count+1

			return pred