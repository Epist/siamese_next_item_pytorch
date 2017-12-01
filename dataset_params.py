#Script for loading precompiled datasets based on parameters
import os.path

def get_dataset_fn(p):
	dataset = p["dataset"]
	if dataset == "amazon_videoGames":
		data_path = "./data/amazon_videogames/data_tables_split_"
	elif dataset == "ml1m":
		data_path = "./data/ml1m/data_tables_split_"
	elif dataset == "beeradvocate":
		data_path = "./data/beeradvocate/data_tables_split_"
	elif dataset == "amazon_automotive":
		data_path = "./data/amazon_automotive/data_tables_split_"
	elif dataset == "amazon_clothing":
		data_path = "./data/amazon_clothing/data_tables_split_"
	elif dataset == "googlelocal":
		data_path = "./data/googlelocal/data_tables_split_"


	trainValidTest = p["train_valid_test"]
	data_path += str(trainValidTest[0]) + "_" + str(trainValidTest[1]) + "_" + str(trainValidTest[2])

	filter_size = p["filter_min"]
	if filter_size is not None:
		data_path += "_filter" + str(filter_size)

	overlap = p["overlap"]
	if overlap:
		data_path += "_withoverlap"

	splittype = p["splittype"]
	if splittype == "transrec":
		data_path += '_transrec'

	if os.path.isfile(data_path+"_metadata.json"):
		return data_path
	else:
		print(data_path)
		raise(Exception("Dataset was not precompiled with the desired parameters"))