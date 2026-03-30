# -*- coding: utf-8 -*-
import re
import os
import json
import torch
import random
import numpy as np
from PIL import Image
from openai import OpenAI
from collections import Counter
from scipy.stats import entropy
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, ViTImageProcessor
from diffusers import StableDiffusionPipeline

def clean_filename(filename):
	filename = re.sub(r'[\\/*?:"<>|]', "", filename)
	return filename


def select_random_indices(input_list, num_refer_cap):
	n = len(input_list)
	if n == 0:
		return []
	if n <= num_refer_cap:
		return list(range(n))

	return random.sample(range(n), 5)

def clean_filename(filename):
	filename = re.sub(r'[\\/*?:"<>|]', "", filename)
	return filename

class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None, augmentation_transform_list=None):
        self.dataset = dataset
        self.transform = transform
        self.augmentation_transform_list = augmentation_transform_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)

        return img, label

    def add_augmented_images(self, augmented_classes_list, label_list):
        self.augmented_data = []

        for indice, l in enumerate(label_list):
            num_augmented_img = augmented_classes_list[indice]
            if num_augmented_img > 0:
                target_images = [idx for idx, (_, label) in enumerate(self.dataset) if label.item() == l]
                if target_images:
                    if len(target_images) >= num_augmented_img:
                        idx_list = random.sample(target_images, num_augmented_img)
                    else:
                        idx_list = random.choices(target_images, k=num_augmented_img)
                    for idx in idx_list:
                        image, label = self.dataset[idx]
                        image = transforms.ToPILImage()(image)
                        selected_augmentation_transform = random.choice(self.augmentation_transform_list)
                        augmentation_transform = transforms.Compose([selected_augmentation_transform, transforms.ToTensor()])
                        augmented_image = augmentation_transform(image)
                        self.augmented_data.append((augmented_image, label))
        self.dataset.extend(self.augmented_data)

def class_loss(labels, losses, num_classes):
	loss_mean = [None for j in range(num_classes)]
	loss_var = [None for j in range(num_classes)]
	for label in set(labels):
		indices = [i for i, x in enumerate(labels) if x == label]
		cls_loss_mean = np.mean(np.array(losses)[indices])
		loss_mean[label] = cls_loss_mean
		cls_var_mean = np.var(np.array(losses)[indices])
		loss_var[label] = cls_var_mean

	filtered_values = [x for x in loss_mean if x is not None]
	mean_value = np.mean(filtered_values)

	loss_info = []
	for l in loss_mean:
		if l is not None:
			if l > mean_value:
				loss_info.append(False)
			else:
				loss_info.append(True)
		else:
			loss_info.append(None)

	return loss_info, loss_mean, loss_var

def list_to_pil_image(image_list):
	images = []
	for np_image in image_list:
		if np_image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
			np_image = np_image.transpose(1, 2, 0)
		image = Image.fromarray((np_image * 255).astype(np.uint8))
		images.append(image)

	return images


def select_element_prob(label_list, num_images):
	label_counts = Counter(label_list)
	total_count = sum(label_counts.values())
	probabilities = {label: count / total_count for label, count in label_counts.items()}
	labels_list = list(probabilities.keys())
	prob_list = list(probabilities.values())
	target_classes_list = np.random.choice(labels_list, size=num_images, p=prob_list, replace=False)

	return target_classes_list


def selected_image(loss_info, min_images, max_images):
	images_dict = {i: None for i in range(len(loss_info))}
	for idx, loss in enumerate(loss_info):
		if loss is False:
			images_dict[idx] = min_images[idx]
		elif loss is True:
			images_dict[idx] = max_images[idx]

	return images_dict


def calculate_JS_div(data_distribution):
    num_classes = len(data_distribution)
    P = np.array(data_distribution) / np.sum(data_distribution)
    Q = np.ones(num_classes) / num_classes
    M = (P + Q) / 2
    j_s_div = 0.5 * entropy(P, M) + 0.5 * entropy(Q, M)
    return j_s_div


def laplace_mechanism(sensitivity, epsilon):
	noise = np.random.laplace(0, sensitivity / epsilon)
	return noise

def add_dp_to_caption(caption, dp_ratio):

	words = caption.split()
	anonymized_words = []

	for word in words:
		dp_value = -1
		while dp_value < 0:
			dp_value = laplace_mechanism(sensitivity=1, epsilon=dp_ratio)

		if dp_value <= 2.0:  # save the word
			anonymized_words.append(word)

	return " ".join(anonymized_words)


def numpy_to_pil_image(image_dict):
	images = []
	idx_list = []
	for idx, i in enumerate(range(len(image_dict))):
		if image_dict[i] is not None:
			idx_list.append(i)
			np_image = image_dict[i]
			if np_image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
				np_image = np_image.transpose(1, 2, 0)
			image = Image.fromarray((np_image * 255).astype(np.uint8))
			# image.save(f"./figures/image_{i}.png")

			images.append(image)



	return images, idx_list


def image_to_text(conf, client_id, image_dict):
	# cap_dp = conf['cap_dp']
	# cap_dp_ratio = conf['cap_dp_ratio']
	data_name = conf['data']


	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	model_name = "Salesforce/blip-image-captioning-large"
	imageprocess = BlipProcessor.from_pretrained(model_name)
	model = BlipForConditionalGeneration.from_pretrained(model_name)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)


	## process numpy image
	images, idx_list = numpy_to_pil_image(image_dict)
	encoding = imageprocess(images=images, return_tensors="pt")
	pixel_values = encoding['pixel_values']
	pixel_values = pixel_values.to(device)

	## image captioning
	max_length = 24
	num_beams = 4
	gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
	output_ids = model.generate(pixel_values, **gen_kwargs)
	captions = imageprocess.batch_decode(output_ids, skip_special_tokens=True)

	save_path = f"./local_data_for_cap/{data_name}/{client_id}"
	os.makedirs(save_path, exist_ok=True)

	return_captions = []
	for idx, caption in enumerate(captions):
		class_idx = idx_list[idx]
		filename = clean_filename(captions[idx]) + f"{class_idx}.png"
		images[idx].save(os.path.join(save_path, filename))
		# if cap_dp == "true":
		# 	caption = add_dp_to_caption(caption, cap_dp_ratio)
		return_captions.append(caption)


	return return_captions

def read_data(dir, idx, is_train=True):
    if is_train:
        data_dir = os.path.join(dir, 'train/')
    else:
        data_dir = os.path.join(dir, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data

def read_client_data(dir, idx, is_train=True):
    data = read_data(dir, idx, is_train)
    X_list = torch.Tensor(data['x']).type(torch.float32)
    y_list = torch.Tensor(data['y']).type(torch.int64)

    client_data = [(x, y) for x, y in zip(X_list, y_list)]
    return client_data

def split_dataset(dataset, split_ratio=0.2):
	dataset_length = len(dataset)
	full_idx = range(dataset_length)
	test_idx = []
	for c in range(len(dataset.classes)):
		target_class_list = [index for (index, value) in enumerate(dataset.targets) if value == c]
		target_class_test_length = int(split_ratio* len(target_class_list))
		test_idx.extend(np.random.permutation(target_class_list)[0:target_class_test_length])
	train_idx = list(set(full_idx).difference(set(test_idx)))

	return train_idx, test_idx


class ImageDatasetFromFileNames(Dataset):
	def __init__(self, fns, labels, transform=None, target_transform=None):
		self.fns = fns
		self.labels = labels
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		x = Image.open(self.fns[index])
		if self.transform is not None:
			x = self.transform(x)
		y = self.labels[index]
		if self.target_transform is not None:
			y = self.target_transform(y)

		return x, y

	def __len__(self):
		return len(self.labels)


class CustomDataset(Dataset):
	def __init__(self, dataset, transform):
		self.dataset = dataset
		self.transform = transform
		self.target_transform = transforms.Compose([transforms.ToTensor()])

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		img_path, label = self.dataset[item]

		img = Image.open(img_path)
		if self.transform is not None:
			img = self.transform(img)
		label = torch.tensor(label, dtype=torch.int64)

		return img, label

	def add_data(self, new_image_list):
		self.dataset.extend(new_image_list)

class AugmentedDataset_v2(Dataset):
	def __init__(self, dataset, augmented_classes_list, label_list, transform=None, augmentation_transform_list=None):
		self.dataset = dataset
		self.transform = transform
		self.augmentation_transform_list = augmentation_transform_list

		self.augmented_data = []
		for indice, l in enumerate(label_list):
			num_augmented_img = augmented_classes_list[indice]
			if num_augmented_img > 0:
				target_images = [idx for idx, (_, label) in enumerate(self.dataset) if label == l]
				if target_images:
					if len(target_images) >= num_augmented_img:
						idx_list = random.sample(target_images, num_augmented_img)
					else:
						idx_list = random.choices(target_images, k=num_augmented_img)
					for idx in idx_list:
						img_path, label = self.dataset[idx]
						img = Image.open(img_path)
						# image = transforms.ToPILImage()(img)
						selected_augmentation_transform = random.choice(self.augmentation_transform_list)
						augmentation_transform = transforms.Compose([selected_augmentation_transform])
						augmented_image = augmentation_transform(img)
						self.augmented_data.append((augmented_image, label))

	def __len__(self):
		return len(self.augmented_data)

	def __getitem__(self, idx):
		img, label = self.augmented_data[idx]
		if self.transform:
			img = self.transform(img)

		return img, label

