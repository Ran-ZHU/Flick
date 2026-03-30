# -*- coding: utf-8 -*-
import torch
import numpy as np
from copy import deepcopy
from Nets.models import get_model
from utils.utils import *
from utils.utils_client import *
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

class Client(object):
	def __init__(self, conf, train_data, id=-1):
		
		self.conf = conf
		self.local_model = get_model(self.conf)
		self.client_id = id
		self.dataset = train_data
		self.train_loader = DataLoader(self.dataset, batch_size=self.conf["batch_size"], drop_last=True, shuffle=True, num_workers=4)

	def update_dataloader(self):
		self.train_loader = DataLoader(self.dataset, batch_size=self.conf["batch_size"], drop_last=True, shuffle=True, num_workers=4)

	def receive_new_data(self, new_images):
		self.dataset.extend(new_images)
		self.update_dataloader()

	def local_train(self, model):
		labels = [y.item() for x, y in self.dataset]
		class_freq = [0 for i in range(self.conf['num_classes'])]
		print('labels:', set(labels))
		for item in set(labels):
			class_freq[item] = labels.count(item)
			print('the %d has found %d' % (item, labels.count(item)))
		print(f"Local data volume: {len(self.dataset)} with {len(set(labels))} classes")

		# load the latest global model
		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())

		# train the model using local data
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], weight_decay=5e-4, momentum=0.9)
		self.local_model.train()
		correct = 0
		train_losses = []
		train_labels = []
		min_losses = {i: float('inf') for i in range(self.conf['num_classes'])}
		max_losses = {i: 0 for i in range(self.conf['num_classes'])}
		min_images = {i: None for i in range(self.conf['num_classes'])}
		max_images = {i: None for i in range(self.conf['num_classes'])}
		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if e == int(self.conf["local_epochs"]-1):
					train_labels.extend(target.cpu().numpy())

				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = cross_entropy(output, target)

				if e == int(self.conf["local_epochs"]-1):
					loss_tmp = cross_entropy(output, target, reduction='none').cpu().detach().numpy()
					train_losses.extend(loss_tmp)
					pred = output.data.max(1)[1]  # get the index of the max log-probability
					correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

					for i in range(data.size(0)):
						y = target[i].item()
						l = loss_tmp[i]
						if l < min_losses[y]:
							min_losses[y] = l
							min_images[y] = data[i].cpu().detach().numpy()
						if l > max_losses[y]:
							max_losses[y] = l
							max_images[y] = data[i].cpu().detach().numpy()

				loss.backward()
				optimizer.step()


		diff = dict()
		local_model_dict = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			local_model_dict[name] = data
		if len(train_losses) > 0:
			train_acc, train_loss = correct/len(train_losses), np.mean(train_losses)
		else:
			train_acc = 0; train_loss= 0
		# print (f"Training Acc: {train_acc}  |  Loss: {train_loss}")

		loss_info, cls_loss_mean, cls_loss_var = class_loss(train_labels, train_losses, self.conf['num_classes'])
		if not any(loss is False for loss in loss_info):
			captions = {}
			print("No captions!!!")
		else:
			selected_images = selected_image(loss_info, min_images, max_images)
			captions = image_to_text(self.conf, self.client_id, selected_images)

		# torch.save(self.local_model.state_dict(), os.path.join("./checkpoints", "localmodel-%d.pkl" % self.client_id))



		return captions, diff, local_model_dict, len(self.dataset)


