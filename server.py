# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import numpy as np
from Nets.models import get_model
from torch.nn.functional import cross_entropy

class Server(object):
	def __init__(self, conf, eval_data_list):
		self.conf = conf
		self.global_model = get_model(self.conf)
		self.hist = None

		self.eval_loader_list = []
		for i in range(len(eval_data_list)):
			self.eval_loader_list.append(DataLoader(eval_data_list[i], batch_size=self.conf["batch_size"]))

	def model_aggregate(self, diff_record):
		total_samples = 0
		samples = []
		for i in range(len(diff_record)):
			total_samples += diff_record[i][1]
			samples.append(diff_record[i][1])
		prob_each_client = np.array(samples) / total_samples
		print ('>>>>prob_each_client:', prob_each_client)

		for num in range(len(diff_record)):
			for name, params in self.global_model.state_dict().items():
				add_item = prob_each_client[num] * diff_record[num][0][name]

				if params.type() != add_item.type():
					params.add_(add_item.to(torch.int64))
				else:
					params.add_(add_item)


	def global_model_fine_tune(self, eval_data_loader):
		# acc_list, loss_list = self.model_eval()
		# print("Before finetuning, Mean acc: %f, Mean loss: %f\n" % (np.mean(acc_list), np.mean(loss_list)))

		print("-----Fine tuning the FC layer-----")
		for name, param in self.global_model.named_parameters():
			if name == "fc.weight" or name == "fc.bias":
				param.requires_grad = True
			else:
				param.requires_grad = False
		# optimizer = optim.SGD(self.global_model.fc.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
		optimizer = torch.optim.Adam(self.global_model.parameters(), lr=1e-3)
		num_epochs = 5
		self.global_model.train()
		for epoch in range(num_epochs):
			running_loss = 0.0
			for inputs, labels in eval_data_loader:
				optimizer.zero_grad()
				if torch.cuda.is_available():
					inputs = inputs.cuda()
					labels = labels.cuda()

				outputs = self.global_model(inputs)
				loss = cross_entropy(outputs, labels)
				loss.backward()
				optimizer.step()
				running_loss += loss.item()
			print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(eval_data_loader):.4f}')

		# acc_list, loss_list = self.model_eval()
		# print("After FC, Mean acc: %f, Mean loss: %f\n" % (np.mean(acc_list), np.mean(loss_list)))

		print("-----Fine tuning ALL layers-----")
		for param in self.global_model.parameters():
			param.requires_grad = True
		# optimizer = optim.SGD(self.global_model.parameters(), lr=1e-4, weight_decay=5e-4, momentum=0.9)
		optimizer = torch.optim.Adam(self.global_model.parameters(), lr=1e-4)
		optimizer.zero_grad()
		num_epochs = 5
		self.global_model.train()
		for epoch in range(num_epochs):
			running_loss = 0.0
			for inputs, labels in eval_data_loader:
				if torch.cuda.is_available():
					inputs = inputs.cuda()
					labels = labels.cuda()
				outputs = self.global_model(inputs)
				loss = cross_entropy(outputs, labels)
				loss.backward()
				optimizer.step()
				running_loss += loss.item()
			print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(eval_data_loader):.4f}')


	def model_eval(self):
		acc_list = []; loss_list = []
		for eval_loader in self.eval_loader_list:
			total_loss = 0.0
			correct = 0
			dataset_size = 0
			self.global_model.eval()
			for batch_id, batch in enumerate(eval_loader):
				data, target = batch
				dataset_size += data.size()[0]

				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()

				output = self.global_model(data)
				total_loss += cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
				pred = output.data.max(1)[1]  # get the index of the max log-probability
				correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

			acc = float(correct) / float(dataset_size)
			total_l = total_loss / dataset_size
			acc_list.append(acc)
			loss_list.append(total_l)

		return acc_list, loss_list