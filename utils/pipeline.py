# -*- coding: utf-8 -*-
import os
import json
import openai
from openai import OpenAI
from time import time
import ast
import re
import itertools
import networkx as nx
import torch
from transformers import BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
import requests
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader


EXAMPLE_MESSAGE = f"""Here is the information:
					Number of generated new image captions: 5
					Provided image captions: ['a cartoon horse with a tongue sticking out of its mouth', 'an image of a painting of a horse running in a field']
					Class name: horse
					"""
EXAMPLE_ANSWER = f"""I should generate 5 new image captions highlighting "horse" class.
					Firstly the possible image domains of the provided image captions could include a cartoon representation and a painting of a horse in a natural setting.
					Then, to create unique new captions while maintaining the "horse" class, I will generate captions that reflect various artistic and cartoon styles, as well as different scenarios where horses can be featured prominently.
					The new captions in JSON format are as follows:
					{{
					"captions": [
									"a cartoon image of a horse wearing a funny hat.",
									"a colorful painting of a horse trotting on the beach at sunset.",
									"a whimsical cartoon of a horse with large, expressive eyes.",
									"an oil painting of a horse jumping over a fence.", 
									"a cartoon of a horse balancing on its hind legs with a surprised look.",
									]
					}}
					"""
SYSTEM_PROMPT = f"""You are a helpful assistant. You utilize the provided image captions to generate new image captions. 
				You should analyze the possible image domains indicated by each provided image caption, then generate new captions that should contain one of these domains'infomation. 
				Besides, new captions should adopt the content shown in the given captions but vary them to ensure uniqueness. 
				Note that the generated captions still feature their class prominently.

				You will be given a given the number of captions that need to be generated and their class name. 
				Also, you will be provided some captions that are from the same class but different images in a classification task.

				Your response should use the following format:
				<reasoning>
				<repeat until you have a decision>
				Output the new captions strictly in the following JSON format:
				{{
				    "captions": [
				        caption1,
				        caption2, 
				        ...
				    ]
				}}
				Only return the JSON output. Do not include any additional text.
					"""

MODEL = "gpt-4o-mini"
# MODEL = "gpt-3.5-turbo-0125"
# MODEL = "gpt-4o"
# MODEL = "gpt-4"

def check_list(n_clients, eval_threshold, acc_list, candidates_idx):
	client_list = []
	if len(acc_list) > 0:
		for i in range(n_clients):
			if acc_list[i] < eval_threshold:
				client_list.append(candidates_idx[i])
	else:
		for i in range(n_clients):
			client_list.append(candidates_idx[i])

	return client_list

def find_captions_by_class(conf, captions, class_name, openai_api_key):
	client = OpenAI(api_key=openai_api_key)
	category_descriptions = get_category_descriptions(conf)
	description = category_descriptions[f"{class_name}"]
	captions_str = "\n".join([f"{caption}" for i, caption in enumerate(captions)])
	prompt = f"""The following is a list of captions. Identify and return only the captions that are related to the category '{class_name}'. The category '{class_name}' is defined as: {description}
	    If none of the captions are related to the category '{class_name}', return an empty list: [].
	    Captions:
	    {captions_str}
	    Output the captions that are related to the category '{class_name}' in valid JSON format, with double quotes around the captions:
	    {{
	    "selected_captions":[
	    caption1, 
	    caption2, 
	    ...,
	    ]
	    }}
	    Only return the JSON output. Do not include any additional text."""

	response = client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}],
		max_tokens=350,
		n=1,
	)
	reply = response.choices[0].message.content
	if reply == "[]":
		return []
	else:
		return reply

def generate_prompt(conf, captions, number_of_captions, class_name):
	category_descriptions = get_category_descriptions(conf)
	description = category_descriptions[f"{class_name}"]
	captions_str = "\n".join([f"{caption}" for i, caption in enumerate(captions)])
	captions_text = "\n".join([f"{i + 1}. \"{caption}\"" for i, caption in enumerate(captions)])
	captions_placeholder = ",\n        ".join([f'"caption{i + 1}"' for i in range(number_of_captions)])

	return f"""
	Given a list of captions:\n {captions_str}, \n select those that are relevant to the main subject:{class_name}. The category '{class_name}' is defined as: {description}.
	Then, analyze the selected captions, which depict various scenes and contexts but consistently center around the main subject: {class_name}.
	Based on this analysis, generate {number_of_captions} new captions that still focus on the main subject of "{class_name}". You may fuse domains, styles, entities, or contexts from the provided captions, but ensure that any new elements introduced do not detract from or obscure the main subject, which must remain the central focus. Introducing new elements is encouraged, but they should enrich the context rather than shift the focus away from the main subject.
	Your response should use the following format:
	<reasoning>
	<repeat until you have a decision>
	Output the {number_of_captions} new captions strictly in the following JSON format:
	{{
	    "captions": [
	        {captions_placeholder}
	    ]
	}}
	Make sure to strictly obey the above format.
	If none of the captions are related to the category '{class_name}', return 
	{{
	"captions": []
	}}
	"""

def img_retrieval(sentence_model, retrieval_threshold, data_pool, provided_captions, label):
	retrieval_image_list = []
	obtain_retrieval_indices_list = []
	caption_list = []
	sim_score_list = []
	ref_cap_find_sim = []
	if len(data_pool) > 0:
		for cap_idx, ref_cap in enumerate(provided_captions):
			max_sim_score = 0
			max_idx = 0
			second_max_sim_score = 0
			second_max_idx = 0
			for idx, sample in enumerate(data_pool):
				sim_score = sentence_calculate_similarity(sentence_model, ref_cap, sample[0])
				if sim_score > max_sim_score:
					second_max_sim_score = max_sim_score
					second_max_idx = max_idx
					max_sim_score = sim_score
					max_idx = idx

			if max_sim_score >= retrieval_threshold:
				print(max_sim_score)
				sim_score_list.append(max_sim_score)
				retrieval_image_list.append((data_pool[max_idx][1], torch.tensor(label, dtype=torch.int64)))
				obtain_retrieval_indices_list.append(cap_idx)
				caption_list.append(data_pool[second_max_idx][0])
				ref_cap_find_sim.append(ref_cap)

		return retrieval_image_list, obtain_retrieval_indices_list, caption_list, sim_score_list, ref_cap_find_sim
	else:
		return [], [], [], [], []

def image_response(prompt, openai_api_key):
	image_generation_client = OpenAI(api_key=openai_api_key)
	try:
		response = image_generation_client.images.generate(
			model="dall-e-2",
			prompt=prompt,
			size="256x256",
			n=1
		)
		image_url = response.data[0].url
		return image_url
	except openai.error.InvalidRequestError as e:
		if e.code == "content_policy_violation":
			print("Content policy violation detected. Adjust your prompt.")
			return False
		else:
			print(f"An error occurred: {e}")

def fetch_image(caption, image_url):
	response = requests.get(image_url)
	if response.status_code == 200:
		image = Image.open(BytesIO(response.content))
		image = image.resize((224, 224))
		return caption, image
	else:
		print(f"Failed to retrieve the image from {image_url}")
		return None

def data_pool_update(tar_caption_list, tar_img_list, size_of_data_pool, data_pool, sentence_model, sim_threshold):
	for i in range(len(tar_img_list)):
		tar_caption = tar_caption_list[i]
		max_score = 0
		max_idx = 0
		for idx, sample in enumerate(data_pool):
			caption, _ = sample
			sim_score = sentence_calculate_similarity(sentence_model, tar_caption, caption)
			if sim_score > max_score:
				max_score = sim_score
				max_idx = idx
		if max_score >= sim_threshold:
			del data_pool[max_idx]
		data_pool.append((tar_caption, tar_img_list[i][0]))

	if len(data_pool) >= size_of_data_pool:
		data_pool = data_pool[-size_of_data_pool:]
	return data_pool

def get_category_descriptions(conf):
	if conf['data'] == "office-caltech":
		category_descriptions = {
			"monitor": "A description involving screens or display devices used to visualize output from computers or other electronic devices, even if 'monitor' is not explicitly mentioned.",
			"projector": "A description involving devices used to project images or videos onto surfaces, typically used in presentations or home theaters, even if 'projector' is not explicitly mentioned.",
			"calculator": "A description involving devices used to perform arithmetic and mathematical calculations, even if 'calculator' is not explicitly mentioned.",
			"bike": "A description involving bicycles or similar pedal-driven vehicles used for transportation or exercise, even if 'bike' or 'bicycle' is not explicitly mentioned.",
			"headphones": "A description involving devices worn over the ears or in-ear to listen to audio privately, even if 'headphones' or 'earphones' are not explicitly mentioned.",
			"mouse": "A description involving handheld input devices used to control a computer cursor, often with buttons and scroll functions, even if 'mouse' is not explicitly mentioned.",
			"mug": "A description involving large cups, typically with handles, used for holding hot beverages such as coffee or tea, even if 'mug' is not explicitly mentioned.",
			"keyboard": "A description involving input devices with a set of keys used for typing text or commands into computers, even if 'keyboard' is not explicitly mentioned.",
			"backpack": "A description involving bags worn on the back, used for carrying books, laptops, or personal items, typically with two straps, even if 'backpack' is not explicitly mentioned.",
			"laptop": "A description involving portable computers with integrated screens and keyboards, used for personal or professional tasks, even if 'laptop' is not explicitly mentioned."
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

	return category_descriptions

def sentence_calculate_similarity(sentence_model, caption1, caption2):
	embedding1 = sentence_model.encode(caption1, convert_to_tensor=True)
	embedding2 = sentence_model.encode(caption2, convert_to_tensor=True)
	similarity = cosine_similarity(embedding1.cpu().numpy().reshape(1, -1), embedding2.cpu().numpy().reshape(1, -1))
	return similarity[0][0]



def get_images(conf, pipe, sentence_model, generated_data_pool, caption_list, all_client_acc_list, class_name_list, candidates_idx):
	data_generation_method = conf['data_generation']
	num_classes = conf['num_classes']
	number_of_captions = conf['data_budget']
	openai_api_key = conf['openai_api_key']

	new_image_dict = {client_id: [] for client_id in candidates_idx}

	tmp_num_generation_img = 0
	tmp_num_retrieval_img = 0
	num_class_count = 0
	for i in range(num_classes):
		if len(all_client_acc_list) != 0:
			acc_list = [all_client_acc_list[j][i] for j in range(conf['k'])]
		else:
			acc_list = []
		client_list = check_list(conf['k'], conf['cls_eval_threshold'], acc_list, candidates_idx)

		if len(client_list) > 0:
			num_class_count += 1
			print("\n")
			print(f"===========Processing the {i}th class============")
			class_name = class_name_list[i]
			raw_selected_captions = find_captions_by_class(conf, caption_list, class_name, openai_api_key)
			match = re.search(r'\{.*\}', raw_selected_captions, re.DOTALL)
			selected_captions = match.group(0)
			try:
				result = json.loads(selected_captions)
				# print (result)
				provided_captions = result.get('selected_captions', [])
			except json.JSONDecodeError:
				print("Error decoding JSON output.")
				provided_captions = []
				continue
			print (f"select the captions related to class {i}:")
			print (provided_captions)

			client = OpenAI(api_key=openai_api_key)

			prompt = generate_prompt(conf, caption_list, number_of_captions, class_name)
			# print ("text prompt: \n")
			# print (prompt)

			response = client.chat.completions.create(
				model=MODEL,
				messages=[
					{"role": "system", "content": "You are a caption generation assistant."},
					{"role": "user", "content": prompt}],
				max_tokens=500,
				n=1,
				stop=None,
				seed=2,
				temperature=0)
			raw_generated_content = response.choices[0].message.content
			# print ("raw_generated_content: \n")
			# print (raw_generated_content)
			match = re.search(r'\{.*\}', raw_generated_content, re.DOTALL)
			generated_content = match.group(0)
			print ("generated captions:")
			print (generated_content)

			try:
				result = json.loads(generated_content)
				list_caption = result.get('captions', [])
				if len(list_caption) == 0:
					continue
			except json.JSONDecodeError:
				print("Error decoding JSON output.")
				list_caption = []
				continue


			# prompt = f"""
			# 		Here is the information:
			# 		Number of generated new image captions: {number_of_captions}
			# 		Provided image captions: {provided_captions}
			# 		Class name: {class_name}
			# 		Only return the JSON output. Do not include any additional text.
			# 		"""
			# response = client.chat.completions.create(
			# 	model=MODEL,
			# 	messages=[
			# 		{"role": "system", "content": SYSTEM_PROMPT},
			# 		{"role": "user", "content": EXAMPLE_MESSAGE},
			# 		{"role": "assistant", "content": EXAMPLE_ANSWER},
			# 		{"role": "user", "content": prompt}],
			# 	max_tokens=500,
			# 	n=1,
			# 	stop=None,
			# 	seed=2,
			# 	temperature=0)
			#
			# # process the caption response
			# raw_generated_content = response.choices[0].message.content
			# print("raw_generated_content: \n")
			# print(raw_generated_content)
			# match = re.search(r'\{.*\}', raw_generated_content, re.DOTALL)
			# generated_content = match.group(0)
			# print(generated_content)
			#
			# try:
			# 	result = json.loads(generated_content)  # 将输出转换为Python字典
			# 	list_caption = result.get('captions', [])  # 从JSON中获取新生成的caption列表
			# 	if len(list_caption) == 0:
			# 		continue
			# except json.JSONDecodeError:
			# 	print("Error decoding JSON output.")
			# 	list_caption = []
			# 	continue


			##  image retrieval


			new_images_list = []
			retrival_images_list, retrival_indices_list, retrival_caption_list, sim_score_list, ref_cap_find_sim = \
				img_retrieval(sentence_model, conf['retrieval_threshold'], generated_data_pool[i], list_caption, i)
			tmp_num_retrieval_img += len(retrival_images_list)

			##  image generation
			print(f"Perform data generation using method {conf['data_generation']}...")
			time_begin_image_generation = time()
			if len(retrival_images_list) < number_of_captions:
				caption_for_generation_list = []
				generation_indices_list = []
				for cap_idx, cap in enumerate(list_caption):
					if cap_idx not in retrival_indices_list:
						caption_for_generation_list.append(cap)
						generation_indices_list.append(cap_idx)

				tmp_num_generation_img += len(caption_for_generation_list)
				tmp_generation_img = []
				if data_generation_method == "df":
					images = pipe(caption_for_generation_list, num_inference_steps=30).images
					for idx, img in enumerate(images):
						img = img.resize((224, 224))
						img = np.array(img)
						img = img.transpose(2, 0, 1)
						tmp_generation_img.append(
							(torch.from_numpy(img / 255).to(torch.float32), torch.tensor(i, dtype=torch.int64)))
				elif data_generation_method == "dall-e-2":
					image_url_list = []
					with ThreadPoolExecutor() as executor:
						future_to_prompt = {executor.submit(image_response, caption, openai_api_key): caption for caption in caption_for_generation_list}
						for future in as_completed(future_to_prompt):
							caption = future_to_prompt[future]
							image_url = future.result()
							if image_url is not False:
								image_url_list.append((caption, image_url))

					# load generated image
					with ThreadPoolExecutor() as executor:
						future_to_url = {executor.submit(fetch_image, caption, url): (caption, url) for caption, url in image_url_list}
						for future in as_completed(future_to_url):
							caption, img = future.result()
							if img is not None:
								img = img.resize((224, 224))
								img = np.array(img)
								img = img.transpose(2, 0, 1)

								tmp_generation_img.append((torch.from_numpy(img / 255).to(torch.float32), torch.tensor(i, dtype=torch.int64)))

				# update data pool
				generated_data_pool[i] = data_pool_update(caption_for_generation_list, tmp_generation_img, conf['size_of_data_pool'],
				                                          generated_data_pool[i], sentence_model, conf['retrieval_threshold'])

			# retrieval imgs + generation imgs
			for j in range(len(list_caption)):
				if j in retrival_indices_list:
					retrieval_idx = retrival_indices_list.index(j)
					new_images_list.append(retrival_images_list[retrieval_idx])
				else:
					generation_idx = generation_indices_list.index(j)
					new_images_list.append(tmp_generation_img[generation_idx])

			# allocate imgs to different clients
			for c_id in client_list:
				new_image_dict[c_id].extend(new_images_list)

	num_images_list = [tmp_num_retrieval_img, tmp_num_generation_img]

	# clustering generated data following a balanced label distribution
	num_cls_sample = np.min([len(generated_data_pool[i]) for i in range(num_classes)])
	fine_tuning_data_list = []
	for i in range(num_classes):
		fine_tuning_data_list.extend(random.sample([(sample[1], torch.tensor(i, dtype=torch.int64)) for sample in generated_data_pool[i]], num_cls_sample))

	return generated_data_pool, new_image_dict, num_images_list, fine_tuning_data_list, num_class_count
