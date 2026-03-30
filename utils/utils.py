# -*- coding: utf-8 -*-
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_other_conf(conf):

	if conf['data'] == "pacs":
		num_classes = 7
	else:
		num_classes = 10

	if conf['data_distribution'] == "natural":
		alpha = None
	elif conf['data_distribution'] == "dirichlet-0.05":
		alpha = 0.05
	elif conf['data_distribution'] == "dirichlet-0.1":
		alpha = 0.1
	elif conf['data_distribution'] == "dirichlet-0.5":
		alpha = 0.5
	elif conf['data_distribution'] == "iid":
		alpha = 10000
	record_file_name = "./outputs/Flick{}_{}_{}_{}.csv".format(conf['flick'], conf['data'], conf['data_distribution'], conf['record_file_num'])

	return num_classes, record_file_name, alpha

def get_category_descriptions(conf):
	if conf['data'] == "office-home":
		category_descriptions = {
			"Desk_Lamp": "A description involving light sources typically used on desks or workspaces to provide focused illumination, even if 'desk lamp' is not explicitly mentioned.",
			"File_Cabinet": "A description involving storage furniture designed for organizing files and documents, even if 'file cabinet' is not explicitly mentioned.",
			"Telephone": "A description involving devices used for voice communication over a distance, even if 'telephone' or 'phone' is not explicitly mentioned.",
			"Bottle": "A description involving containers for liquids, typically with narrow necks, even if 'bottle' is not explicitly mentioned.",
			"Soda": "A description involving carbonated soft drinks, often sweetened, even if 'soda' is not explicitly mentioned.",
			"Calendar": "A description involving tools for tracking dates, months, and events, even if 'calendar' is not explicitly mentioned.",
			"TV": "A description involving devices or screens used for watching broadcasts or digital media, even if 'TV' or 'television' is not explicitly mentioned.",
			"Alarm_Clock": "A description involving devices used to wake people up at specific times, even if 'alarm clock' is not explicitly mentioned.",
			"Fork": "A description involving utensils with prongs used for eating or serving food, even if 'fork' is not explicitly mentioned.",
			"Bucket": "A description involving containers with open tops used to hold or transport liquids or other materials, even if 'bucket' is not explicitly mentioned.",
			"Sink": "A description involving basins used for washing hands, dishes, or other objects, even if 'sink' is not explicitly mentioned.",
			"Hammer": "A description involving tools with a heavy head used for pounding nails or breaking objects, even if 'hammer' is not explicitly mentioned.",
			"Ruler": "A description involving straight-edged tools used for measuring length, even if 'ruler' is not explicitly mentioned.",
			"Mop": "A description involving tools used for cleaning floors, typically with long handles and absorbent heads, even if 'mop' is not explicitly mentioned.",
			"Spoon": "A description involving utensils with a shallow bowl used for eating or serving food, even if 'spoon' is not explicitly mentioned.",
			"Paper_Clip": "A description involving small metal or plastic devices used to hold sheets of paper together, even if 'paper clip' is not explicitly mentioned.",
			"Clipboards": "A description involving flat boards with clips for holding papers steady while writing, even if 'clipboard' is not explicitly mentioned.",
			"Refrigerator": "A description involving appliances used to keep food and beverages cold, even if 'refrigerator' is not explicitly mentioned.",
			"Couch": "A description involving seating furniture, typically upholstered, used for relaxing, even if 'couch' or 'sofa' is not explicitly mentioned.",
			"Chair": "A description involving seating furniture designed for one person, even if 'chair' is not explicitly mentioned.",
			"Webcam": "A description involving cameras connected to computers for recording or streaming video, even if 'webcam' is not explicitly mentioned.",
			"Folder": "A description involving tools used to organize and store papers, typically with pockets or tabs, even if 'folder' is not explicitly mentioned.",
			"ToothBrush": "A description involving tools used for cleaning teeth, even if 'toothbrush' is not explicitly mentioned.",
			"Pan": "A description involving flat-bottomed cooking vessels, typically used for frying or sautéing, even if 'pan' is not explicitly mentioned.",
			"Kettle": "A description involving containers, typically with a spout and handle, used for boiling water, even if 'kettle' is not explicitly mentioned.",
			"Marker": "A description involving writing tools with broad tips for marking or highlighting, even if 'marker' is not explicitly mentioned.",
			"Lamp_Shade": "A description involving coverings for lamps, used to diffuse light, even if 'lamp shade' is not explicitly mentioned.",
			"Monitor": "A description involving screens used to display output from computers, even if 'monitor' is not explicitly mentioned.",
			"Curtains": "A description involving fabric coverings used to block light or provide privacy at windows, even if 'curtains' are not explicitly mentioned.",
			"Mug": "A description involving large cups, typically with handles, used for hot beverages, even if 'mug' is not explicitly mentioned.",
			"Radio": "A description involving devices used to receive and play audio broadcasts, even if 'radio' is not explicitly mentioned.",
			"Bed": "A description involving furniture used for sleeping or resting, even if 'bed' is not explicitly mentioned.",
			"Calculator": "A description involving devices used to perform mathematical calculations, even if 'calculator' is not explicitly mentioned.",
			"Postit_Notes": "A description involving small sticky notes used for reminders or temporary messages, even if 'Post-it notes' are not explicitly mentioned.",
			"Glasses": "A description involving eyeglasses used for vision correction, even if 'glasses' are not explicitly mentioned.",
			"Sneakers": "A description involving athletic shoes designed for comfort and performance, even if 'sneakers' are not explicitly mentioned.",
			"Flowers": "A description involving blooming plants or floral arrangements, even if 'flowers' are not explicitly mentioned.",
			"Keyboard": "A description involving input devices used to type on computers, even if 'keyboard' is not explicitly mentioned.",
			"Shelf": "A description involving flat surfaces used for storing or displaying items, even if 'shelf' is not explicitly mentioned.",
			"Laptop": "A description involving portable computers with screens and keyboards, even if 'laptop' is not explicitly mentioned.",
			"Helmet": "A description involving protective headgear used for safety in various activities, even if 'helmet' is not explicitly mentioned.",
			"Table": "A description involving flat surfaces used for working or eating, typically supported by legs, even if 'table' is not explicitly mentioned.",
			"Mouse": "A description involving input devices used to control a computer cursor, even if 'mouse' is not explicitly mentioned.",
			"Notebook": "A description involving books with blank or lined pages used for writing or taking notes, even if 'notebook' is not explicitly mentioned.",
			"Exit_Sign": "A description involving signs used to indicate the location of exits in buildings, even if 'exit sign' is not explicitly mentioned.",
			"Drill": "A description involving tools used to bore holes or fasten screws, even if 'drill' is not explicitly mentioned.",
			"Bike": "A description involving bicycles or similar pedal-driven vehicles, even if 'bike' or 'bicycle' is not explicitly mentioned.",
			"Push_Pin": "A description involving small pins used for attaching paper to boards or walls, even if 'push pin' is not explicitly mentioned.",
			"Computer": "A description involving electronic devices used for processing and storing data, even if 'computer' is not explicitly mentioned.",
			"Screwdriver": "A description involving tools used for turning screws, even if 'screwdriver' is not explicitly mentioned.",
			"Eraser": "A description involving tools used to remove pencil marks, even if 'eraser' is not explicitly mentioned.",
			"Toys": "A description involving objects used for play, especially by children, even if 'toys' are not explicitly mentioned.",
			"Backpack": "A description involving bags worn on the back, typically used for carrying books or personal items, even if 'backpack' is not explicitly mentioned.",
			"Pen": "A description involving writing instruments that use ink, even if 'pen' is not explicitly mentioned.",
			"Batteries": "A description involving devices used to store and supply electrical energy, even if 'batteries' are not explicitly mentioned.",
			"Printer": "A description involving devices used to produce physical copies of digital documents, even if 'printer' is not explicitly mentioned.",
			"Pencil": "A description involving writing instruments made of graphite or other materials, even if 'pencil' is not explicitly mentioned.",
			"Candles": "A description involving objects made of wax with a wick, used for light or decoration, even if 'candles' are not explicitly mentioned.",
			"Scissors": "A description involving tools with two blades used for cutting, even if 'scissors' are not explicitly mentioned.",
			"Knives": "A description involving sharp tools used for cutting food or materials, even if 'knives' are not explicitly mentioned.",
			"Trash_Can": "A description involving containers used for holding waste, even if 'trash can' is not explicitly mentioned.",
			"Flipflops": "A description involving casual footwear with a Y-shaped strap, even if 'flipflops' are not explicitly mentioned.",
			"Oven": "A description involving appliances used for baking or roasting food, even if 'oven' is not explicitly mentioned.",
			"Fan": "A description involving devices used to circulate air or cool spaces, even if 'fan' is not explicitly mentioned.",
			"Speaker": "A description involving devices used to project sound, even if 'speaker' is not explicitly mentioned."
		}

	elif conf['data'] == "pacs":
		category_descriptions = {
			"person": "A description involving people, children, adults, or groups of humans, even if they are not explicitly mentioned as 'person' or 'people'. This includes activities like standing, sitting, walking, or being in a group.",
			"dog": "A description involving dogs or canines, even if the word 'dog' is not mentioned. This includes activities such as running, barking, playing, or interacting with people.",
			"house": "A description involving buildings, homes, or structures typically used for living, even if they are not explicitly mentioned as 'house' or 'building'.",
			"horse": "A description involving horses, equestrian activities, or animals similar to horses, even if 'horse' is not explicitly mentioned.",
			"elephant": "A description involving elephants or large land mammals with trunks, even if 'elephant' is not explicitly mentioned.",
			"giraffe": "A description involving giraffes or long-necked animals typically found in the wild, even if 'giraffe' is not explicitly mentioned.",
			"guitar": "A description involving guitars or musical instruments with strings that are played by strumming or plucking, even if the word 'guitar' is not explicitly mentioned."
		}

	elif conf['data'] == "domainnet":
		category_descriptions = {
			"airplane": "A description involving aircraft, planes, jets, or flying vehicles, even if the word 'airplane' is not explicitly mentioned. This includes scenes like flying in the sky, landing on runways, or being at an airport.",
			"clock": "A description involving timekeeping devices such as wall clocks, alarm clocks, or watches, even if the word 'clock' is not explicitly used. This includes references to telling time, hands moving, or faces with numbers.",
			"axe": "A description involving axes or tools used for chopping wood or similar tasks, even if 'axe' is not directly mentioned. This includes scenes of cutting, logging, or handling heavy-bladed tools.",
			"basketball": "A description involving basketballs, basketball games, or related activities, even if the word 'basketball' is not used. This includes playing on a court, shooting hoops, dribbling, or wearing jerseys.",
			"bicycle": "A description involving bicycles, cycling, or two-wheeled transport, even if 'bicycle' is not explicitly mentioned. This includes riding on roads, pedaling, or bike-related gear.",
			"bird": "A description involving birds or flying animals with feathers, even if the word 'bird' is not explicitly mentioned. This includes chirping, flying, perching, or nests.",
			"strawberry": "A description involving strawberries or small red fruits with seeds on the surface, even if 'strawberry' is not directly mentioned. This includes references to sweet, red fruit in desserts, smoothies, or natural settings.",
			"flower": "A description involving flowers or blooming plants, even if the word 'flower' is not used. This includes petals, stems, floral arrangements, or gardens.",
			"pizza": "A description involving pizza or similar flat, baked dishes with toppings, even if the word 'pizza' is not explicitly mentioned. This includes references to cheese, slices, crusts, or being served in boxes.",
			"bracelet": "A description involving bracelets or wrist accessories, even if 'bracelet' is not explicitly mentioned. This includes references to jewelry worn on the wrist, beaded designs, or decorative bands."
		}
	return category_descriptions

def get_immediate_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_dataset_one_domain(dir, domain, idx2classname, n_classes):
	x_fns = []
	ys = []
	stats_dict = {}
	stats_list = []
	for i in range(n_classes):
		class_name = idx2classname[i]
		x_for_class = list(glob.glob(os.path.join(dir, domain, class_name, "*.jpg"))) + list(
			glob.glob(os.path.join(dir, domain, class_name, "*.png"))) + list(glob.glob(os.path.join(dir, domain, class_name, "*.jpeg")))
		x_fns += x_for_class
		ys += [i for _ in range(len(x_for_class))]
		stats_dict[str(i)] = len(x_for_class)
		stats_list.append(len(x_for_class))
	return np.array(x_fns), np.array(ys), stats_dict, np.array(stats_list)


#### DomainNet dataset
def read_domainnet_data(dir, domain_name, split="train", labels=None):
	data_paths = []
	data_labels = []
	stats_dict = {f"{i}": 0 for i in range(len(labels))}

	split_file = os.path.join(dir, "{}_{}.txt".format(domain_name, split))
	with open(split_file, "r") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			relative_data_path, label = line.split(' ')
			absolute_data_path = os.path.join(dir, relative_data_path)
			label = int(label)
			if labels is not None:
				if label in labels:
					data_paths.append(absolute_data_path)
					data_labels.append(labels.index(label))
					stats_dict[str(labels.index(label))] += 1
			elif labels is None:
				continue

	return data_paths, data_labels, stats_dict

def get_dataset_domainnet(dir, domain_name, if_train=True, labels=None):
	if if_train:
		split = "train"
	else:
		split = "test"
	data_paths, data_labels, stats_dict = read_domainnet_data(dir, domain_name, split=split, labels=labels)

	return data_paths, data_labels, stats_dict

class DomainNet(Dataset):
	def __init__(self, data_paths, data_labels, transforms):
		super(DomainNet, self).__init__()
		self.data_paths = data_paths
		self.data_labels = data_labels
		self.transforms = transforms

	def __getitem__(self, index):
		img = Image.open(self.data_paths[index])
		if not img.mode == "RGB":
			img = img.convert("RGB")
		label = self.data_labels[index]
		img = self.transforms(img)

		return img, label

	def __len__(self):
		return len(self.data_paths)


def involved_client_table(conf):
	np.random.seed(333)
	table = []
	for i in range(conf['global_epochs']):
		table.append(np.random.choice(np.arange(conf['num_clients']), size=conf['k'], replace=False))

	return table



def dirichletSplit(alpha, n_clients, n_classes):
	np.random.seed(222)
	return np.random.dirichlet(n_clients * [alpha], n_classes)


def split_dirichlet(labels, num_clients, alpha):
	np.random.seed(222)
	label_distribution = np.random.dirichlet([alpha] * num_clients, len(labels))
	client_data = [[] for _ in range(num_clients)]

	for i, dist in enumerate(label_distribution):
		for j, frac in enumerate(dist):
			client_data[j].extend(labels[i][:int(frac * len(labels[i]))])
	return client_data


def split_train_test(clt_x_fns, clt_ys, test_portion):
	try:
		X_train_fns, X_test_fns, y_train, y_test = train_test_split(
			clt_x_fns, clt_ys, test_size=test_portion, random_state=333, stratify=clt_ys)
	except ValueError:
		X_train_fns, X_test_fns, y_train, y_test = train_test_split(
			clt_x_fns, clt_ys, test_size=test_portion, random_state=333)

	return X_train_fns, X_test_fns, y_train, y_test

def split2clientsofficehome(x_fns, ys, stats, partitions, client_idx_offset=0, verbose=True):
	print("==> splitting dataset into clients' own datasets")
	n_classes, n_clients = partitions.shape
	splits = []  # n_classes * n_clients
	for i in range(n_classes):
		indices = np.where(ys == i)[0]
		np.random.shuffle(indices)
		cuts = np.cumsum(np.round_(partitions[i] * stats[str(i)]).astype(int))
		cuts = np.clip(cuts, 0, stats[str(i)])
		cuts[-1] = stats[str(i)]
		splits.append(np.split(indices, cuts))

	clients = []
	for i in range(n_clients):
		indices = np.concatenate([splits[j][i] for j in range(n_classes)], axis=0)
		dset = [x_fns[indices], ys[indices]]
		clients.append(dset)
		if verbose:
			print("\tclient %03d has" % (client_idx_offset + i + 1), len(dset[0]), "images")
	return clients


def obtain_augmented_info(augmented_client):
	num_image_list = []
	target_classes_list = []

	for i in range(len(augmented_client)):
		if augmented_client[i] != 0:
			num_image_list.append(augmented_client[i])
			target_classes_list.append(i)

	return num_image_list, target_classes_list
