"""
Data reader

Constructs an ordinal input vector and a one-hot target vector based on a pre-ranked data dictionary.

A next item is chosen at random and the previous N items are pulled from the dictionary and assigned their relative rankings with respect to the next item.

Not sure yet how I want to handle cases where there are fewer than N previously purchased items. Either drop the cases or find a clean way to represent this.

Splits are handled either using online splitting or by loading a split-file that contains the row number assignments. 
Random number generator seeds will be represented explicitly and stored as a property of the reader.
"""

import numpy as np
import json
import timeit
import scipy
import torch
from torch.autograd import Variable
import pandas as pd

class data_reader(object):
	def __init__(self, filename, train_subset_size, valid_subset_size, test_subset_size):
		self.data_fn_root = filename
		self.metadata_fn = self.data_fn_root + "_metadata.json"
		self.train_fn = self.data_fn_root + "_train.json"
		self.valid_fn = self.data_fn_root + "_valid.json"
		self.test_fn = self.data_fn_root + "_test.json"

		with open(self.metadata_fn, "r") as f:
			[self.user_id_dict, self.item_id_dict] = json.load(f)
		self.num_users = len(self.user_id_dict.keys())
		self.num_items = len(self.item_id_dict.keys())
		print("Loaded metadata for ", self.num_users, " users and ", self.num_items, " items from ", self.metadata_fn)

		self.train_data = self.load_data(self.train_fn)
		self.valid_data = self.load_data(self.valid_fn)
		self.test_data = self.load_data(self.test_fn)
		
		self.num_train_ratings = len(self.train_data)
		self.num_valid_ratings = len(self.valid_data)
		self.num_test_ratings = len(self.test_data)

		self.train_subset_size = train_subset_size
		self.valid_subset_size = valid_subset_size
		self.test_subset_size = test_subset_size
		self.num_train_ratings_subset = self.get_subset_size(self.train_data, train_subset_size)
		self.num_valid_ratings_subset = self.get_subset_size(self.valid_data, valid_subset_size)
		self.num_test_ratings_subset = self.get_subset_size(self.test_data, test_subset_size)
		#self.num_train_ratings_subset = self.num_train_ratings
		#self.num_valid_ratings_subset = self.num_valid_ratings
		#self.num_test_ratings_subset = self.num_test_ratings


	def load_data(self, filename):
		# Load dataset and decoding dictionaries
		#return pd.read_json(filename)
		print("Loading ordering data from ", filename)
		with open(filename, "r") as f:
			data = json.load(f)
		return data

	def invert_id_dictionary(self, id_dictionary):
		#Function to create an inverse mapping to the original item or user ids for the purposes of identifying which items are being recommended to wihch users
		pass
		#return inverted_dictionary

	def create_subset_index(self, data_table, n):
		subset_indices = []
		for i, row in enumerate(data_table):
			if row[-1]>=n:
				subset_indices.append(i)
		print("Using datapoints with at least ", n, " previous item(s). This leaves ", len(subset_indices), "/", len(data_table), " datapoints.")
		return subset_indices

	def get_subset_size(self, data_table, n):
		subset_size = 0
		for i, row in enumerate(data_table):
			if row[-1]>=n:
				subset_size += 1
		return subset_size

	def data_gen(self, batch_size, train_valid_test, num_previous_items, use_masking, use_gpu):
		# A generator for batches for the model.
		# A datapoint has the format [ith_item_purchased, i-1th_item_purchased, ..., user, candidate next item 1, candidate next item 2]
		# Where one of the candidate next items is the real next item that the user purchased and the other is an item drawn randomly from the set of all items \ the real next item
		# The distractor item can be an item that the user has never seen, an item they purchased previously, an item that they will purchase later, or even the current item (repeats are allowed)
		
		#The target is a number, either -1 or 1 that represents which item was the true next purchase. If the first item is the next purchase, the target is -1, otherwise it is 1.
		
		if train_valid_test == "train":
			data_table = self.train_data
			subset_size = self.train_subset_size
		elif train_valid_test == "valid":
			data_table = self.valid_data
			subset_size = self.valid_subset_size
		elif train_valid_test == "test":
			data_table = self.test_data
			subset_size = self.test_subset_size
		num_ratings = len(data_table)
		print(train_valid_test, " dataset has " , num_ratings, " ratings")
		#items_list = [list(self.item_id_dict.keys())

		if subset_size:
			subset_indices = self.create_subset_index(data_table, subset_size)
			num_applicable_ratings = len(subset_indices)
		else:
			subset_indices = range(num_ratings)
			num_applicable_ratings = num_ratings

		sun_will_rise_tomorrow = True #Assumption
		while sun_will_rise_tomorrow:

			epoch_order = np.random.permutation(subset_indices)

			cur_position = 0
			while cur_position < num_applicable_ratings:

				if num_applicable_ratings-cur_position>batch_size:
					cur_batch_size = batch_size
				else:
					cur_batch_size = num_applicable_ratings-cur_position

				batch_user_inputs = []
				batch_prev_item_inputs_list = []
				batch_prev_item_masks_list = []
				for i in range(num_previous_items):
					batch_prev_item_inputs_list.append([])
					batch_prev_item_masks_list.append([])
				batch_left_cand_inputs = []
				batch_right_cand_inputs = []
				
				targets = np.zeros([cur_batch_size, 1])

				for datapoint in range(cur_batch_size):

					#Pick a random row from the dataframe

					cur_row_index = epoch_order[cur_position]
					cur_row = data_table[cur_row_index]
					#cur_row = data_table.sample(n=1, replace=True).values[0]

					#Transform the row into a list and append to the batch
					#Handle cases where there are not enough past items
						#Have two cases where you can either return lists with None when there are not enough past items or sample only from the subset of items with enough previous items
						#To do the latter, we can have an extra column in the DF that contains the number of valid previous items. We can the sample fro mthe subset of the DF with nprev>=n_desired_prev
						#The former will be useful for a more flexible model class than we currently have...

					user_info = int(cur_row[0])

					next_item = int(cur_row[1])

					prev_items_infos = []
					prev_item_observed_mask = []
					for j in range(num_previous_items): #The order here is n-1th, n-2th, n-3th, etc.
						if cur_row[j+2] is not None:
							cur_prev_item = int(cur_row[j+2])
							cur_mask_val = 1
						else:
							cur_prev_item = self.num_items #special number representing no input
							cur_mask_val = 0
						prev_items_infos.append(cur_prev_item)
						prev_item_observed_mask.append(cur_mask_val)
					"""
					for item in cur_row[2:]:
						cur_prev_item = int(item)
						prev_items_infos.append(cur_prev_item)
					"""

					#Pick a random item for the contrast proportionally to the frequency of purchase (to speed up training since more frequent items are more difficult)
					while True: #To make sure it is not the identical item...
						#distractor_item_key = np.random.randint(self.num_items)
						#distractor_item = int(items_list[distractor_item_key])
						#distractor_item_key = np.random.choice(list(self.item_id_dict.keys()))
						#distractor_item = int(self.item_id_dict[distractor_item_key])
						#print("Old distractor item: ", distractor_item)
						distractor_item = np.random.randint(self.num_items)
						#print("New distractor item: ", distractor_item)
						if distractor_item != next_item:
							break


					#Choose a random order of presentation for the target and distractor items
					#order = np.random.randint(2)
					#if order == 0:
					left_item_info = next_item
					right_item_info = distractor_item
					targets[datapoint,0] = 1 #True next item is on the left
					#elif order == 1:
					#	left_item_info = distractor_item
					#	right_item_info = next_item
					#	targets[datapoint,0] = -1 #True next item is on the right

					batch_user_inputs.append(user_info)
					#print(user_info)
					#print(prev_items_infos)
					#print(left_item_info)
					#print(right_item_info)
					#print("\n\n")
					#print(prev_items_infos)
					[batch_prev_item_inputs_list[index].append(x) for index, x in enumerate(prev_items_infos)]
					if use_masking:
						[batch_prev_item_masks_list[index].append(x) for index, x in enumerate(prev_item_observed_mask)]
					batch_left_cand_inputs.append(left_item_info)
					batch_right_cand_inputs.append(right_item_info)

					cur_position += 1


				if use_gpu:
					user_inputs_var = Variable(torch.LongTensor(batch_user_inputs)).cuda()
					left_cand_item_inputs_var = Variable(torch.LongTensor(batch_left_cand_inputs)).cuda()
					right_cand_item_inputs_var = Variable(torch.LongTensor(batch_right_cand_inputs)).cuda()
					prev_items_inputs_vars = [Variable(torch.LongTensor(prev_item_input)).cuda() for prev_item_input in batch_prev_item_inputs_list]
					targets_var = Variable(torch.FloatTensor(targets), requires_grad=False).cuda()
					if use_masking:
						prev_items_masks_vars = [Variable(torch.FloatTensor(prev_item_mask)).cuda() for prev_item_mask in batch_prev_item_masks_list]
						yield [[user_inputs_var] + [left_cand_item_inputs_var] + [right_cand_item_inputs_var] + prev_items_inputs_vars + prev_items_masks_vars, targets_var]
					else:
						yield [[user_inputs_var] + [left_cand_item_inputs_var] + [right_cand_item_inputs_var] + prev_items_inputs_vars, targets_var]
				else:
					user_inputs_var = Variable(torch.LongTensor(batch_user_inputs))
					left_cand_item_inputs_var = Variable(torch.LongTensor(batch_left_cand_inputs))
					right_cand_item_inputs_var = Variable(torch.LongTensor(batch_right_cand_inputs))
					prev_items_inputs_vars = [Variable(torch.LongTensor(prev_item_input)) for prev_item_input in batch_prev_item_inputs_list]
					targets_var = Variable(torch.FloatTensor(targets), requires_grad=False)
					if use_masking:
						prev_items_masks_vars = [Variable(torch.FloatTensor(prev_item_mask)) for prev_item_mask in batch_prev_item_masks_list]
						yield [[user_inputs_var] + [left_cand_item_inputs_var] + [right_cand_item_inputs_var] + prev_items_inputs_vars + prev_items_masks_vars, targets_var]
					else:
						yield [[user_inputs_var] + [left_cand_item_inputs_var] + [right_cand_item_inputs_var] + prev_items_inputs_vars, targets_var]

