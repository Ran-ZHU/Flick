# -*- coding: utf-8 -*-
import os
import random
from collections import defaultdict
import torch
import torch.nn as nn
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch.optim as optim
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.utils_client import class_loss
from sentence_transformers import SentenceTransformer
import itertools
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy

sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def sentence_calculate_similarity(caption1, caption2):
	embedding1 = sentence_model.encode(caption1, convert_to_tensor=True)
	embedding2 = sentence_model.encode(caption2, convert_to_tensor=True)
	similarity = cosine_similarity(embedding1.cpu().numpy().reshape(1, -1), embedding2.cpu().numpy().reshape(1, -1))
	return similarity[0][0]

def obtain_matrix(conf, record_dict):
	matrix = [[None for j in range(conf['num_classes'])] for i in range(conf['k'])]
	for idx, dict in enumerate(record_dict):
		for k in list(dict.keys()):
			matrix[idx][k] = dict[k]
	return matrix

def find_indices(matrix):
    num_rows = len(matrix) # 4
    num_cols = len(matrix[0]) #7

    result_dict = {}
    for row in range(num_rows):
        result = []
        for col in range(num_cols):
            result_tmp = []
            if matrix[row][col] is not True:
                result_tmp.append((row, col))
                for other_row in range(num_rows):
                    if other_row != row and matrix[other_row][col] is True:
                        result_tmp.append((other_row, col))

            result.append(result_tmp)
        result_dict[row] = result
    return result_dict

def find_all_indices(matrix):
    num_rows = len(matrix) # 4
    num_cols = len(matrix[0]) #7

    result_dict = {}
    for row in range(num_rows):
        result = []
        for col in range(num_cols):
            result_tmp = []
            result_tmp.append((row, col))
            for other_row in range(num_rows):
                if other_row != row and matrix[other_row][col] is True:
                    result_tmp.append((other_row, col))

            result.append(result_tmp)
        result_dict[row] = result
    return result_dict

def check_list(eval_threshold, acc_list, loss_info):
	cls_list = []
	if len(acc_list) > 0:
		for i in range(len(loss_info)):
			if acc_list[i] < eval_threshold:
				cls_list.append(i)
	else:
		for i in range(len(loss_info)):
			if loss_info[i] is not True:
				cls_list.append(i)

	return cls_list

def put_all_caption_into_list(caption_clients):
	uploaded_caption_list = []
	for i in range(len(caption_clients)):
		caption_list = caption_clients[i]
		for cap in caption_list:
			if cap is not None:
				if cap not in uploaded_caption_list:
					uploaded_caption_list.append(cap)
	return uploaded_caption_list

def aggregate_delta_control(diff_record):
	total_samples = 0
	samples = []
	for i in range(len(diff_record)):
		total_samples += diff_record[i][1]
		samples.append(diff_record[i][1])
	prob_each_client = np.array(samples) / total_samples
	print('>>>>prob_each_client:', prob_each_client)

	control_update = deepcopy(diff_record[0][0])
	for w in control_update:
		control_update[w] = 0.0

	for num in range(len(diff_record)):
		delta_control = diff_record[num][0]
		for w in delta_control:
			control_update[w] += prob_each_client[num] * delta_control[w]

	return control_update

def init_prev_grads(model):
	prev_grads = None
	for param in model.parameters():
		if not isinstance(prev_grads, torch.Tensor):
			prev_grads = torch.zeros_like(param.view(-1))
		else:
			prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
	return prev_grads



def load_data_pool(conf, sub_rate):
	print ("The length of data pool:")
	generated_data_pool = {i: [] for i in range(conf["num_classes"])}
	for i in range(conf["num_classes"]):
		path = f"./new_data/{conf['data']}/{i}/"
		if os.path.exists(path):
			img_name_list = os.listdir(path)
			data_pool_length = int(len(img_name_list)*sub_rate)
			img_name_list = img_name_list[:data_pool_length]
			for img_name in img_name_list:
				filepath = os.path.join(path, img_name)
				try:
					img = Image.open(filepath)
					img = np.array(img)
					img = img.transpose(2, 0, 1)
					file_base_name = os.path.splitext(img_name)[0]
					generated_data_pool[i].append((file_base_name, torch.from_numpy(img/255).to(torch.float32)))
				except Exception as e:
					print(f"Failed to open {img_name}: {e}")
	print ([len(generated_data_pool[i]) for i in range(conf["num_classes"])])
	return generated_data_pool

def load_data_pool_path(conf, sub_rate):
	generated_data_pool = {i: [] for i in range(conf["num_classes"])}
	for i in range(conf["num_classes"]):
		path = f"./new_data/{conf['data']}/{i}/"
		if os.path.exists(path):
			img_name_list = os.listdir(path)
			data_pool_length = int(len(img_name_list)*sub_rate)
			img_name_list = img_name_list[:data_pool_length]
			for img_name in img_name_list:
				img_path = os.path.join(path, img_name)
				file_base_name = os.path.splitext(img_name)[0]
				generated_data_pool[i].append((file_base_name, img_path))
	return generated_data_pool


def test_local_model(conf, tmp_model, local_model_list, generated_data_pool):
	# clustering all generated data
	eval_data_list = []
	all_client_acc_list = []
	num_cls_sample = np.min([len(generated_data_pool[i]) for i in range(conf["num_classes"])])
	if num_cls_sample > 0:
		for i in range(conf["num_classes"]):
			eval_data_list.extend(random.sample([(sample[1], torch.tensor(i, dtype=torch.int64)) for sample in generated_data_pool[i]], num_cls_sample))
		eval_data_loader = DataLoader(eval_data_list, batch_size=conf['batch_size'] * 4, shuffle=False)
		for idx, local_model in enumerate(local_model_list):
			for name, param in tmp_model.state_dict().items(): #
				tmp_model.state_dict()[name].copy_(local_model[name])

			correct = 0
			test_losses = []
			test_acc = {i: 0 for i in range(conf['num_classes'])}
			test_labels = []
			tmp_model.eval()
			for batch_id, batch in enumerate(eval_data_loader):
				data, target = batch
				test_labels.extend(target.cpu().numpy())

				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()

				output = tmp_model(data)
				loss_tmp = cross_entropy(output, target, reduction='none').cpu().detach().numpy()
				test_losses.extend(loss_tmp)
				pred = output.data.max(1)[1]  # get the index of the max log-probability
				correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
				for label, prediction in zip(target, pred):
					if label == prediction:
						test_acc[label.item()] += 1
			print ("The average acc:", correct / len(test_labels))
			print (len(test_labels))
			print ("The class acc:", [test_acc[i]/num_cls_sample for i in range(conf['num_classes'])])
			all_client_acc_list.append([test_acc[i]/num_cls_sample for i in range(conf['num_classes'])])

	return all_client_acc_list

def eval_model(eval_loader, tmp_model):
	class_correct = defaultdict(int)
	class_total = defaultdict(int)
	class_loss = defaultdict(float)
	tmp_model.eval()
	with torch.no_grad():
		for batch_id, batch in enumerate(eval_loader):
			data, target = batch

			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()

			output = tmp_model(data)
			loss = cross_entropy(output, target)
			_, predicted = torch.max(output, 1)
			c = (predicted == target).squeeze()
			for i in range(len(target)):
				label = target[i].item()
				class_correct[label] += c[i].item()
				class_total[label] += 1
				class_loss[label] += loss.item()

	class_correct_list = []; class_loss_list = []
	for i in range(len(class_correct)):
		if class_total[i] > 0:
			class_correct_list.append(class_correct[i] / class_total[i])
			class_loss_list.append(class_loss[i] / class_total[i])
			print(f'Class {i} - Accuracy: {class_correct[i] / class_total[i]:.4f}')
			print(f'Class {i} - Loss: {class_loss[i] / class_total[i]:.4f}')
		else:
			class_correct_list.append(None)
			class_loss_list.append(None)
			print(f'Class {i} - No samples')

	return class_correct_list, class_loss_list


def consistency_mask(epoch, threshold, increase_history, client_id, update_diff):
	updates = update_diff
	if len(increase_history[client_id]) == 0:
		increase_history[client_id] = {key: torch.zeros_like(val) for key, val in updates.items()}

		for key in updates:
			increase_history[client_id][key] = (updates[key] >= 0).float()

		return increase_history, {key: torch.ones_like(val) for key, val in updates.items()}

	mask = {}
	for key in updates:
		positive_consistency = increase_history[client_id][key]
		negative_consistency = 1 - increase_history[client_id][key]
		consistency = torch.where(updates[key] >= 0, positive_consistency, negative_consistency)
		mask[key] = (consistency > threshold).float()

	for key in updates:
		increase = (updates[key] >= 0).float()
		increase_history[client_id][key] = (increase_history[client_id][key] * epoch + increase) / (epoch + 1)

	return increase_history, mask

def compute_distance(update_diff, param_names):
	euclidean_distance = 0
	for key in update_diff.keys():
		if key in param_names:
			euclidean_distance += torch.norm(update_diff[key].float()).item()

	return euclidean_distance


def get_params_diff_weights(candidates_idx, previous_weights, previous_delta_weights, euclidean_distance_dict, beta):
	weight_dict = {}
	total_weight = 0
	for client_id in candidates_idx:
		client_distance = euclidean_distance_dict[client_id]

		delta_weight = (1 - beta) * (previous_delta_weights[client_id]) + beta * (
					(client_distance) / sum(euclidean_distance_dict.values()))

		new_weight = previous_weights[client_id] + delta_weight

		previous_weights[client_id] = new_weight
		previous_delta_weights[client_id] = delta_weight

		weight_dict[client_id] = new_weight
		total_weight += new_weight

	for client_id in candidates_idx:
		weight_dict[client_id] /= total_weight

	return previous_weights, previous_delta_weights, weight_dict

def plot_acc_loss(conf, acc_list, loss_list):
	# obtain the global model acc and loss
	global_mean_acc_list = [np.mean(i) for i in acc_list]
	global_mean_loss_list = [np.mean(i) for i in loss_list]

	domain_acc_list = []
	domain_loss_list = []
	for i in range(int(len(acc_list[0])/5)):
		acc_tmp = np.array(acc_list)[:, i*5:(i+1)*5]
		loss_tmp = np.array(loss_list)[:, i * 5:(i + 1) * 5]
		domain_acc = [np.mean(acc_tmp[j, :]) for j in range(len(acc_list))]
		domain_loss = [np.mean(loss_tmp[j, :]) for j in range(len(loss_tmp))]
		domain_acc_list.append(domain_acc)
		domain_loss_list.append(domain_loss)

	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.plot(np.arange(1, len(global_mean_acc_list)+1), global_mean_acc_list, color="orange", marker="D", markersize=6, label="global model")

	plt.xlabel('Round')
	plt.ylabel('Accuracy')
	plt.title(f"{conf['data']}-{conf['record_file_num']}")
	plt.legend()
	plt.tick_params(labelsize=12)

	plt.subplot(1, 2, 2)
	plt.plot(np.arange(1, len(global_mean_acc_list)+1), global_mean_loss_list, color="orange", marker="D", markersize=6, label="global model")
	plt.xlabel('Round')
	plt.ylabel('Loss')
	plt.tick_params(labelsize=12)
	plt.savefig("./figures/acc_loss_flick{}_{}_{}_{}.jpg".format(conf['flick'], conf['data'], conf['data_distribution'], conf['record_file_num']))
	plt.close()





