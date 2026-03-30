# -*- coding: utf-8 -*-
"""
@Time: 18/07/2024 22:14
@Author: Ran Zhu
@Email: ranzhuzr@gmail.com
@IDE: PyCharm
"""
import os
import subprocess
import time




def run_commands_with_configs(config_files, gpu_id):
	for config in config_files:
		# command = f"python3 main_v6.py -c ./utils/{config}"
		command = f"CUDA_VISIBLE_DEVICES={gpu_id} python3 main.py -c ./utils/{config}"
		print(f"Running command: {command}")
		result = subprocess.run(command, shell=True)

		if result.returncode != 0:
			print(f"Error executing command with config: {config}")
			continue



configs = ["conf.json", ]

gpu_id = 0
run_commands_with_configs(configs, gpu_id=gpu_id)