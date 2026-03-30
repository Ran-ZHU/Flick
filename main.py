# -*- coding: utf-8 -*-
import math
import sys
import torch
import pickle
import pandas as pd
import argparse, json
from time import time
from copy import deepcopy
from torch import nn, Tensor
from torchsummary import summary
from Nets.models import get_model
from utils.get_dataset import *
from utils.utils_server import *
from utils.utils import *
from utils.pipeline import get_images
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from sentence_transformers import SentenceTransformer
from server import Server
from client import Client

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()

	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)

	conf['num_classes'], conf['record_file_name'], conf['alpha'] = get_other_conf(conf)

	# load DF model
	# model_id = "Salesforce/stable-diffusion-1.5"
	model_id = "Jiali/stable-diffusion-1.5"
	pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16).to("cuda")

	# load sentence bert model
	sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to("cuda")

	candidate_table = involved_client_table(conf)  # load candidates involved in each round
	generated_data_pool = {i: [] for i in range(conf["num_classes"])}
	tmp_model = get_model(conf)
	# summary(tmp_model, (3, 224, 224))

	############################################################
	#       Load dataset & Instantiating Server/Clients        #
	############################################################
	transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
	class_name_list, train_data_list, eval_data_list = get_dataset(f"./data/", conf)

	server = Server(conf, eval_data_list)
	clients = []
	for c in range(conf["num_clients"]):
		clients.append(Client(conf, train_data_list[c], c))
	print("\n\n")


	record_transferred_data_idx = {i: {j: [] for j in range(conf["num_classes"])} for i in range(conf["num_clients"])}
	info_record = []
	test_acc_list = []; test_loss_list = []  # global model test info
	training_log_all_rounds = []  # local training log
	tmp_record_clients = []
	client_data_length = {i: [] for i in range(conf["num_clients"])}
	for e in range(conf["global_epochs"]):
		time_0 = time()
		print ("===============Epoch {}==============".format(e+1))
		candidates_idx = candidate_table[e]
		print ("Involved clients:", candidates_idx)
		candidates = [clients[i] for i in candidates_idx]
		# num_generation_dict = {i: {j: conf['data_budget'] for j in range(conf['num_classes'])} for i in range(conf['k'])}
		diff_record = {}
		diff_list = []
		loss_clients = []
		caption_clients = []
		local_model_list = []
		for idx, c in enumerate(candidates):
			client_id = candidates_idx[idx]
			print(">>>Client %d begin training>>>" % client_id)
			captions, diff, local_model_dict, local_data_length = c.local_train(server.global_model)
			diff_record[idx] = (diff, local_data_length)
			local_model_list.append(local_model_dict)
			caption_clients.append(captions)
		server.model_aggregate(diff_record)
		if conf["flick"] == "true":
			############################################################
			#          test local model using generated data           #
			############################################################
			print ("===========Test local model===========")
			all_client_acc_list = test_local_model(conf, tmp_model, local_model_list, generated_data_pool)


			############################################################
			#                      image generation                    #
			############################################################
			caption_list = put_all_caption_into_list(caption_clients)

			generated_data_pool, new_image_dict, num_images_list, fine_tuning_data_list, num_class_count \
				= get_images(
				conf, pipe, sentence_model,
				generated_data_pool,
				caption_list,
				all_client_acc_list,
				class_name_list,
				candidates_idx)
			eval_data_length = len(fine_tuning_data_list)
			if eval_data_length == 0:
				eval_data_loader = DataLoader(fine_tuning_data_list, batch_size=int(conf['data_budget']*conf['num_classes']), shuffle=False)
			else:
				eval_data_loader = DataLoader(fine_tuning_data_list, batch_size=len(fine_tuning_data_list), shuffle=False)

			# transfer generated data to clients
			for idx, c in enumerate(candidates):
				client_id = candidates_idx[idx]
				c.receive_new_data(new_image_dict[client_id])

			############################################################
			#                 aggregation and evaluation               #
			############################################################

			num_fine_tune = 0
			if num_class_count > 0:
				class_correct_list, _ = eval_model(eval_data_loader, server.global_model)
				print ("mean acc:", np.mean(class_correct_list))
				if len(eval_data_loader) > 0:
					server.global_model_fine_tune(eval_data_loader)


		# evaluate global model
		acc_list, loss_list = server.model_eval()
		test_acc_list.append(acc_list)
		test_loss_list.append(loss_list)
		print("Epoch %d on all domains, Mean acc: %f, Mean loss: %f\n" % (e+1, np.mean(acc_list), np.mean(loss_list)))

		############################################################
		#                        Training log                      #
		############################################################
		info_record.append(
			[e + 1, np.mean(acc_list), np.mean(loss_list), conf['flick'], conf['data'], conf['data_distribution'],
			 conf['num_clients'], conf['k'], conf['local_epochs'], conf['batch_size'], conf['lr'],
			 conf['cls_eval_threshold'], conf['data_budget'],
			 conf['data_generation'], conf['size_of_data_pool'],
			 conf['retrieval_threshold']])

		if (e+1) % 2 == 0:
			plot_acc_loss(conf, test_acc_list, test_loss_list)
			pd.DataFrame(info_record).to_csv(conf['record_file_name'], index=False,
			                                 header=['Round', 'Mean_test_acc', 'Mean_test_loss', 'Flick', 'Dataset',
			                                         'Distribution', 'Num_clients', 'K', 'Local_epochs',
			                                         'Batch_szie', 'Lr',
			                                         'cls_eval_threshold',
			                                         'data_budget', 'Data_generation',
			                                         'size_of_data_pool',  "Retrieval_threshold"])


