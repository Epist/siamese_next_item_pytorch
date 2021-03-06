#Generate n-back data tables for next item recommendation
#Needs to be run with python 2


#Also write a data reader class for sampling these and sendign them to the model


from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import json


#Source data parameters
datapath = "/data1/amazon/productGraph/categoryFiles/ratings_Electronics.csv" #"/data1/movielens/ml-1m/ratings.csv" #"/data1/amazon/productGraph/categoryFiles/ratings_Video_Games.csv" #"/data1/googlelocal/googlelocal_ratings_timestamps.csv"  "/data1/beer/beeradvocate-crawler/ba_ratings.csv"
header = False #Needed for beeradvocate dataset

#Dataset generation parameters
save_filename = "data/amazon_electronics/data_tables_split_80_10_10_"
split_ratio = [0.8,0.1,0.1] #Not relevant in transrec mode
n = 5 #Number of previous items to include in each table entry (linearly increases the size of the table)
min_num_ratings = 5
filter_type = "concurrent" #"concurrent" "user-first" "item-first" "just_users" "just_items" "max_k-core"
use_overlapping_intervals = True #Not relevant in transrec mode
splittype = "transrec"  # "percentage" or "transrec"

save_filename += filter_type
save_filename += "filter"
save_filename += str(min_num_ratings)
if use_overlapping_intervals:
	save_filename += "_withoverlap"
if splittype == "transrec":
	save_filename += '_transrec'

print("Loading raw data from ", datapath)
if header:
	raw_data = pd.read_csv(datapath)
else:
	raw_data = pd.read_csv(datapath, header=None)
raw_data.columns = ["userId", "itemId", "rating", "timestamp"]

"""def remove_underrepresented(data, min_num_ratings):
	orig_len = len(data)
	g = data.groupby('userId')
	data = g.filter(lambda x: len(x) >= min_num_ratings)
	g = data.groupby('itemId')
	data = g.filter(lambda x: len(x) >= min_num_ratings)
	new_len = len(data)
	print("Filter kept ", new_len, "/", orig_len, " observations")
	return data"""

def remove_underrepresented(data, min_num_ratings, filter_type):
	orig_len = len(data)

	if filter_type == "concurrent":
		vc_u = data.userId.value_counts()
		vc_i = data.itemId.value_counts()

		u_list = vc_u.index[vc_u.values > min_num_ratings]
		i_list = vc_i.index[vc_i.values > min_num_ratings]

		data = data[data.userId.isin(u_list)]
		data = data[data.itemId.isin(i_list)]
	elif filter_type == "max_k-core":
		print("Searching for max k-core starting from k =",min_num_ratings)
		k=min_num_ratings
		while k>0:
			d = data
			last_len = len(d) + 1
			while last_len > len(d):
				#print("iteration ", i, "  last len: ", last_len)
				last_len = len(d)
				vc_u = d.userId.value_counts()
				vc_i = d.itemId.value_counts()

				u_list = vc_u.index[vc_u.values > k]
				i_list = vc_i.index[vc_i.values > k]

				d = d[d.userId.isin(u_list)]
				d = d[d.itemId.isin(i_list)]
			if len(d)>0:
				data = d
				break
			else:
				print("Data is ",str(k)+"-degenerate")
				k -= 1
		print("Found a max k core of k =",k)

	elif filter_type == "user_first":
		vc_u = data.userId.value_counts()
		u_list = vc_u.index[vc_u.values > min_num_ratings]
		data = data[data.userId.isin(u_list)]

		vc_i = data.itemId.value_counts()
		i_list = vc_i.index[vc_i.values > min_num_ratings]
		data = data[data.itemId.isin(i_list)]
	elif filter_type == "item_first":
		vc_i = data.itemId.value_counts()
		i_list = vc_i.index[vc_i.values > min_num_ratings]
		data = data[data.itemId.isin(i_list)]

		vc_u = data.userId.value_counts()
		u_list = vc_u.index[vc_u.values > min_num_ratings]
		data = data[data.userId.isin(u_list)]
	elif filter_type == "just_users":
		vc_u = data.userId.value_counts()
		u_list = vc_u.index[vc_u.values > min_num_ratings]
		data = data[data.userId.isin(u_list)]
	elif filter_type == "just_items":
		vc_i = data.itemId.value_counts()
		i_list = vc_i.index[vc_i.values > min_num_ratings]
		data = data[data.itemId.isin(i_list)]
	else:
		raise(Exception("Filter type not supported"))

	new_len = len(data)
	print("Filter kept ", new_len, "/", orig_len, " observations")
	return data

def generate_id_dict(ratings, ID_type):
	id_dict = {}
	Ids = []	
	old_ids = list(ratings[ID_type])	
	current_id_num = 0
	for str_id in old_ids:
		if str_id not in id_dict:
			id_dict[str_id] = str(current_id_num)
			current_id_num += 1   
	if ID_type=="userId":
		print("Assigned IDs to ", current_id_num, " unique users")
	elif ID_type=="itemId":
		print("Assigned IDs to ", current_id_num, " unique items")
	return id_dict

def build_user_item_dict(data, user_id_dict, item_id_dict):
	#Group item and time of purchase data by user
	user_dict_timestamps = {}
	for i, row in data.iterrows():
		raw_user = row["userId"]
		user = user_id_dict[raw_user]
		raw_item = row["itemId"]
		item = item_id_dict[raw_item]
		timestamp = str(row["timestamp"])
		if user in user_dict_timestamps:
			user_dict_timestamps[user].append((item, timestamp))
		else:
			user_dict_timestamps[user] = [(item, timestamp)]
	#Change the timestamps into ordinal rankings and sort by the ranking
	user_item_orders = {}
	for user in user_dict_timestamps:
		item_timestamp_list = user_dict_timestamps[user]
		user_item_orders[user] = sort_and_rank(item_timestamp_list)
	return user_item_orders

def sort_and_rank(item_timestamp_list):
	sorted_item_timestamp_list = sorted(item_timestamp_list, key=lambda t: int(float(t[1])))
	sorted_item_list = [x[0] for i,x in enumerate(sorted_item_timestamp_list)]
	return sorted_item_list

if min_num_ratings>0:
	print("Filtering data")
	filtered_data = remove_underrepresented(raw_data, min_num_ratings, filter_type)
else:
	filtered_data = raw_data
	
print("Building user dict")
user_id_dict = generate_id_dict(filtered_data, "userId")
print("Building item dict")
item_id_dict = generate_id_dict(filtered_data, "itemId")

print("Joining, sorting, and ranking")
user_item_dict = build_user_item_dict(filtered_data, user_id_dict, item_id_dict)


def split_by_userwise_percentage(user_item_dict, train_valid_test_split, n, use_overlap=False):
	train_percentage = train_valid_test_split[0]
	valid_percentage = train_valid_test_split[1]
	test_percentage = train_valid_test_split[2]
	if train_percentage+valid_percentage+test_percentage != 1:
		raise(exception("Split percentages must add up to 1."))
		
	train_dict = {}
	valid_dict = {}
	test_dict = {}
	
	for user in user_item_dict:
		purchase_list = user_item_dict[user]
		num_items = len(purchase_list)
		train_dict[user] = purchase_list[0:int(np.ceil(train_percentage*num_items))]

		if use_overlap:
			valid_dict[user] = purchase_list[int(np.ceil(train_percentage*num_items))-n:int(np.ceil((train_percentage+valid_percentage)*num_items))]
			test_dict[user] = purchase_list[int(np.ceil((train_percentage+valid_percentage)*num_items))-n:]
		else:
			valid_dict[user] = purchase_list[int(np.ceil(train_percentage*num_items)):int(np.ceil((train_percentage+valid_percentage)*num_items))]
			test_dict[user] = purchase_list[int(np.ceil((train_percentage+valid_percentage)*num_items)):]
	
	return [train_dict, valid_dict, test_dict]

def split_by_transrec_paper(user_item_dict, n):
	train_dict = {}
	valid_dict = {}
	test_dict = {}
	
	for user in user_item_dict:
		purchase_list = user_item_dict[user]

		train_dict[user] = purchase_list[0:-2]
		valid_dict[user] = purchase_list[-2-n:-1]
		test_dict[user] = purchase_list[-1-n:]
	
	return [train_dict, valid_dict, test_dict]

print("Splitting into training, validation, and test data")

if splittype == "percentage":
	train_dict, valid_dict, test_dict = split_by_userwise_percentage(user_item_dict, split_ratio, n, use_overlap=use_overlapping_intervals)
elif splittype == "transrec":
	train_dict, valid_dict, test_dict = split_by_transrec_paper(user_item_dict, n)

def gen_nback_table(data, n, use_overlap=False):

	if use_overlap: #Do not include items from the previous datasets as next items
		start_index = n
	else:
		start_index = 0
	data_table = [] #Rows are [userID, curItem, 1-backItem, 2-backItem, etc.]
	for user in data:
		user_item_list = data[user]

		for i in range(start_index, len(user_item_list)):
			cur_row = [user]
			num_prev = -1
			for j in range(n+1):
				if i-j >= 0:
					cur_row.append(user_item_list[i-j])
					num_prev += 1
				else:
					cur_row.append(None)
			cur_row.append(num_prev) #Additional 
			data_table.append(cur_row)
	#pandas_data = pd.DataFrame(data_table)
	#colnames = ["userId", "nextItemId"]
	#colnames.extend(["n_minus_"+str(i)+"_ItemID" for i in range(1,n+1)])
	#colnames.append("num_prev")
	#pandas_data.columns = colnames
	#return pandas_data
	return data_table

def gen_nback_table_transrec_paper(data, n):

	#ui_list_lens = []
	data_table = [] #Rows are [userID, curItem, 1-backItem, 2-backItem, etc.]
	for user in data:
		user_item_list = data[user]

		cur_row = [user]
		num_prev = -1
		num_obs = len(user_item_list)
		for j in range(n+1):
			if num_obs-j > 0:
				cur_row.append(user_item_list[num_obs-j-1])
				num_prev += 1
			else:
				cur_row.append(None)
		cur_row.append(num_prev) #Additional 
		data_table.append(cur_row)
	#print("Set: ", list(set(ui_list_lens)))
	return data_table

print("Generating train data table")
train_table = gen_nback_table(train_dict, n, use_overlap=False) #Training data never has overlap
if splittype == "percentage":
	print("Generating validation data table")
	valid_table = gen_nback_table(valid_dict, n, use_overlap=use_overlapping_intervals)
	print("Generating test data table")
	test_table  = gen_nback_table(test_dict, n, use_overlap=use_overlapping_intervals)
elif splittype=="transrec":
	print("Generating validation data table")
	valid_table = gen_nback_table_transrec_paper(valid_dict, n)
	print("Generating test data table")
	test_table  = gen_nback_table_transrec_paper(test_dict, n)


#train_table.to_json(save_filename+"_train.json")
#valid_table.to_json(save_filename+"_valid.json")
#test_table.to_json(save_filename+"_test.json")

print("Saving data")
with open(save_filename+"_train.json", "w") as f:
	json.dump(train_table, f)
with open(save_filename+"_valid.json", "w") as f:
	json.dump(valid_table, f)
with open(save_filename+"_test.json", "w") as f:
	json.dump(test_table, f)


#Save metadata
metadata = [user_id_dict, item_id_dict]

with open(save_filename+"_metadata.json", "w") as f:
	json.dump(metadata, f)

print("Data saved as: ", save_filename)

