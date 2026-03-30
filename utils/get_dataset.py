# -*- coding: utf-8 -*-
import os
import glob
import pickle
import itertools
from torch.utils.data import ConcatDataset, DataLoader, Subset
from utils.utils import *
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from utils.utils_client import ImageDatasetFromFileNames, read_client_data, split_dataset

def get_dataset(dir, conf):
	name, distribution, num_clients, alpha = conf["data"], conf["data_distribution"], conf["num_clients"], conf['alpha']

	if name == 'office-caltech':
		raw_data_path = os.path.join(dir, "office_caltech_10/")
		domains = ['amazon', 'caltech', 'dslr', 'webcam']
		transform = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')), transforms.Resize((224, 224)), transforms.ToTensor()])
		save_data_path = os.path.join(dir, f"./processed_Officecaltech/Caltech{num_clients}-{distribution}")
		if not os.path.exists(save_data_path):
			os.makedirs(save_data_path)

		tmp_class_name_list = []
		for fn in get_immediate_subdirectories(raw_data_path + domains[0]):
			tmp_class_name_list.append(os.path.split(fn)[1])
		class_name_list = sorted(tmp_class_name_list)
		print (class_name_list)
		# class_name_list = ['monitor', 'projector', 'calculator', 'bike', 'headphones', 'mouse', 'mug', 'keyboard', 'backpack', 'laptop']
		assert num_clients % 4 == 0
		N_CLIENTS_PER_DOMAIN = num_clients // len(domains)
		idx2classname = {i: c_name for i, c_name in enumerate(class_name_list)}
		get_class_folder = lambda domain, class_n: os.path.join(raw_data_path, domain, class_n)
		n_classes = len(class_name_list)
		if not os.path.exists(os.path.join(save_data_path, "train/")):
			clients = []
			for domain_idx, domain in enumerate(domains):
				dataset_domain_fns, dataset_domain_labels, dataset_stats_dict, dataset_stats_list = get_dataset_one_domain(
					raw_data_path, domain, idx2classname, n_classes)
				partitions = np.zeros((n_classes, N_CLIENTS_PER_DOMAIN))
				partitions = dirichletSplit(alpha=alpha, n_clients=N_CLIENTS_PER_DOMAIN, n_classes=n_classes)

				clients += split2clientsofficehome(dataset_domain_fns,
				                                   dataset_domain_labels,
				                                   dataset_stats_dict,
				                                   partitions,
				                                   client_idx_offset=domain_idx * N_CLIENTS_PER_DOMAIN,
				                                   verbose=True)

			os.makedirs(os.path.join(save_data_path, "train/"))
			os.makedirs(os.path.join(save_data_path, "test/"))
			train_data_list = []; eval_data_list = []
			for client_idx, (clt_x_fns, clt_ys) in enumerate(clients):
				X_train_fns, X_test_fns, y_train, y_test = split_train_test(clt_x_fns, clt_ys, test_portion=0.2)
				train_set = ImageDatasetFromFileNames(X_train_fns, y_train, transform=transform)
				test_set = ImageDatasetFromFileNames(X_test_fns, y_test, transform=transform)

				train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
				test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

				xs_train, ys_train = next(iter(train_loader))
				xs_test, ys_test = next(iter(test_loader))

				train_dict = {"x": xs_train.numpy(), "y": ys_train.numpy()}
				test_dict = {"x": xs_test.numpy(), "y": ys_test.numpy()}
				for data_dict, npz_fn in [(train_dict, os.path.join(save_data_path, "train", f"{client_idx}.npz")),
				                          (test_dict, os.path.join(save_data_path, "test", f"{client_idx}.npz"))]:
					with open(npz_fn, "wb") as f:
						np.savez_compressed(f, data=data_dict)


				print("------------")
				train_tmp = [(x, y) for x, y in zip(xs_train.type(torch.float32), ys_train.type(torch.int64))]
				test_tmp = [(x, y) for x, y in zip(xs_test.type(torch.float32), ys_test.type(torch.int64))]
				train_data_list.append(train_tmp)
				eval_data_list.append(test_tmp)
			print(">>>>>>>Finished generating the dataset")
		else:
			train_data_list = []
			tmp_eval_data_list = []
			for i in range(num_clients):
				train_data = read_client_data(save_data_path, i, is_train=True)
				test_data = read_client_data(save_data_path, i, is_train=False)
				train_data_list.append(train_data)
				tmp_eval_data_list.append(test_data)

			eval_data_list = []
			for i in range(0, num_clients, N_CLIENTS_PER_DOMAIN):
				combined_list = []
				for sublist in tmp_eval_data_list[i:i + N_CLIENTS_PER_DOMAIN]:
					combined_list.extend(sublist)
				eval_data_list.append(combined_list)
			tmp_eval_data_list = []

			print(">>>>>>>Finished loading the dataset")

		return class_name_list, train_data_list, eval_data_list

	elif name == 'pacs':
		domains = ['art_painting', 'cartoon', 'photo', 'sketch']
		transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
		save_data_path = os.path.join(dir, f"./processed_PACS/PACS{num_clients}-{distribution}")
		raw_data_path = os.path.join(dir, "PACS/")
		tmp_class_name_list = []
		for fn in get_immediate_subdirectories(raw_data_path + domains[0]):
			tmp_class_name_list.append(os.path.split(fn)[1])
		class_name_list = sorted(tmp_class_name_list)
		print(class_name_list)
		# class_name_list = ["dog", "person", "giraffe", "guitar", "house", "horse", "elephant"]
		assert num_clients % 4 == 0
		N_CLIENTS_PER_DOMAIN = num_clients // len(domains)
		idx2classname = {i: c_name for i, c_name in enumerate(class_name_list)}
		get_class_folder = lambda domain, class_n: os.path.join(raw_data_path, domain, class_n)
		n_classes = len(class_name_list)
		if not os.path.exists(os.path.join(save_data_path, "train/")):
			clients = []
			for domain_idx, domain in enumerate(domains):
				dataset_domain_fns, dataset_domain_labels, dataset_stats_dict, dataset_stats_list = get_dataset_one_domain(
					raw_data_path, domain, idx2classname, n_classes)
				partitions = np.zeros((n_classes, N_CLIENTS_PER_DOMAIN))
				partitions = dirichletSplit(alpha=alpha, n_clients=N_CLIENTS_PER_DOMAIN, n_classes=n_classes)

				clients += split2clientsofficehome(dataset_domain_fns,
				                                   dataset_domain_labels,
				                                   dataset_stats_dict,
				                                   partitions,
				                                   client_idx_offset=domain_idx * N_CLIENTS_PER_DOMAIN,
				                                   verbose=True)

			os.makedirs(os.path.join(save_data_path, "train/"))
			os.makedirs(os.path.join(save_data_path, "test/"))
			train_data_list = []; eval_data_list = []
			for client_idx, (clt_x_fns, clt_ys) in enumerate(clients):
				X_train_fns, X_test_fns, y_train, y_test = split_train_test(clt_x_fns, clt_ys, test_portion=0.2)
				train_set = ImageDatasetFromFileNames(X_train_fns, y_train, transform=transform)
				test_set = ImageDatasetFromFileNames(X_test_fns, y_test, transform=transform)

				train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
				test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

				xs_train, ys_train = next(iter(train_loader))
				xs_test, ys_test = next(iter(test_loader))

				train_dict = {"x": xs_train.numpy(), "y": ys_train.numpy()}
				test_dict = {"x": xs_test.numpy(), "y": ys_test.numpy()}
				for data_dict, npz_fn in [(train_dict, os.path.join(save_data_path, "train", f"{client_idx}.npz")),
				                          (test_dict, os.path.join(save_data_path, "test", f"{client_idx}.npz"))]:
					with open(npz_fn, "wb") as f:
						np.savez_compressed(f, data=data_dict)

				print("------------")
				train_tmp = [(x, y) for x, y in zip(xs_train.type(torch.float32), ys_train.type(torch.int64))]
				test_tmp = [(x, y) for x, y in zip(xs_test.type(torch.float32), ys_test.type(torch.int64))]
				train_data_list.append(train_tmp)
				eval_data_list.append(test_tmp)
			print(">>>>>>>Finished generating the dataset")

		else:
			train_data_list = []
			tmp_eval_data_list = []
			for i in range(num_clients):
				train_data = read_client_data(save_data_path, i, is_train=True)
				test_data = read_client_data(save_data_path, i, is_train=False)
				train_data_list.append(train_data)
				tmp_eval_data_list.append(test_data)

			eval_data_list = []
			for i in range(0, num_clients, N_CLIENTS_PER_DOMAIN):
				combined_list = []
				for sublist in tmp_eval_data_list[i:i+N_CLIENTS_PER_DOMAIN]:
					combined_list.extend(sublist)
				eval_data_list.append(combined_list)
			tmp_eval_data_list = []

			print(">>>>>>>Finished loading the dataset")

		return class_name_list, train_data_list, eval_data_list


	elif name == "domainnet":

		domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
		labels = [1, 73, 11, 19, 29, 31, 290, 121, 225, 39]
		transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
		save_data_path = os.path.join(dir, f"./data/domainnet{num_clients}-{distribution}")

		raw_data_path = os.path.join("./data/domainnet/")
		class_name_list = ["airplane", "clock", "axe", "basketball", "bicycle", "bird", "strawberry", "flower", "pizza",
		                   "bracelet"]

		n_classes = len(class_name_list)
		if not os.path.exists(os.path.join(save_data_path, "train/")):
			os.makedirs(os.path.join(save_data_path, "train/"))
			os.makedirs(os.path.join(save_data_path, "test/"))

			train_data_list = []
			eval_data_list = []
			clients = []
			test_data = []
			for domain_idx, domain in enumerate(domains):
				print("Domain:", domain)
				if domain == "quickdraw":
					N_CLIENTS_PER_DOMAIN = 15
				else:
					N_CLIENTS_PER_DOMAIN = 17
				print(N_CLIENTS_PER_DOMAIN)
				dataset_domain_fns, dataset_domain_labels, dataset_stats_dict = get_dataset_domainnet(raw_data_path,
				                                                                                      domain,
				                                                                                      if_train=True,
				                                                                                      labels=labels)
				print(dataset_stats_dict)
				testdata_domain_fns, testdata_domain_labels, testdata_stats_dict = get_dataset_domainnet(raw_data_path,
				                                                                                         domain,
				                                                                                         if_train=False,
				                                                                                         labels=labels)
				test_data.append((testdata_domain_fns, testdata_domain_labels))

				partitions = dirichletSplit(alpha=alpha, n_clients=N_CLIENTS_PER_DOMAIN, n_classes=n_classes)
				clients += split2clientsofficehome(np.array(dataset_domain_fns), np.array(dataset_domain_labels),
				                                   dataset_stats_dict, partitions, client_idx_offset=len(clients), verbose=True)

			### generate training data
			for client_idx, (clt_x_fns, clt_ys) in enumerate(clients):
				# client_dataset = DomainNet(clt_x_fns, clt_ys, transform)
				# train_data.append(client_dataset)

				train_set = ImageDatasetFromFileNames(clt_x_fns, clt_ys, transform=transform)
				train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
				xs_train, ys_train = next(iter(train_loader))
				train_dict = {"x": xs_train.numpy(), "y": ys_train.numpy()}

				for data_dict, npz_fn in [(train_dict, os.path.join(save_data_path, "train", f"{client_idx}.npz"))]:
					with open(npz_fn, "wb") as f:
						np.savez_compressed(f, data=data_dict)
				print("------------")

				train_tmp = [(x, y) for x, y in zip(xs_train.type(torch.float32), ys_train.type(torch.int64))]
				train_data_list.append(train_tmp)

			### generate test data

			for domain_idx, (X_test_fns, y_test) in enumerate(test_data):
				test_set = ImageDatasetFromFileNames(X_test_fns, y_test, transform=transform)
				test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
				xs_test, ys_test = next(iter(test_loader))
				test_dict = {"x": xs_test.numpy(), "y": ys_test.numpy()}

				for data_dict, npz_fn in [
					(test_dict, os.path.join(save_data_path, "test", f"domain_{domain_idx}.npz"))]:
					with open(npz_fn, "wb") as f:
						np.savez_compressed(f, data=data_dict)
				print("------------")

				test_tmp = [(x, y) for x, y in zip(xs_test.type(torch.float32), ys_test.type(torch.int64))]
				eval_data_list.append(test_tmp)
			print(">>>>>>>Finished generating the dataset")

		else:
			train_data_list = []
			for i in range(num_clients):
				train_data = read_client_data(save_data_path, i, is_train=True)
				train_data_list.append(train_data)

			eval_data_list = []
			for i in range(len(domains)):
				test_data = read_client_data(save_data_path, i, is_train=False, is_domainnet=True)
				eval_data_list.append(test_data)
			print(">>>>>>>Finished loading the dataset")

		return train_data_list, eval_data_list, class_name_list




