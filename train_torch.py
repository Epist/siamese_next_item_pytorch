"""
Recommending next item from last N items using neural networks

Represents last N in terms of ordinal rank in their item entry in the input vector. Predicts a probability distribution over next items. 

Use Python 3
"""
from data_reader_torch_tables import data_reader
from model_torch import SiameseRecNet
from model_torch_shared_embeddings import SiameseRecNet as SiameseRecNet_shared
import pandas as pd
import numpy as np
import datetime
import torch
import copy
import math

#Parameters:

#Dataset parameters 
dataset = "beeradvocate" # movielens20m, amazon_books, amazon_moviesAndTv, amazon_videoGames, amazon_clothing, beeradvocate, yelp, netflix, ml1m, amazon_automotive
subset_size = None

#Training parameters
max_epochs = 100
batch_size = 64
patience = 5
early_stopping_metric = "mae"

#Model parameters
numlayers = 2
num_hidden_units = 128
embedding_size = 128
num_previous_items = 2
model_save_path = "models/"
model_loss = 'mse'
optimizer_type = 'rmsprop'
activation_type = 'tanh'
use_shared_embedding_model = True
l2_regularization = 0
dropout_prob = 0.5

model_save_name = "next_item_prediction_"+str(batch_size)+"bs_"+str(numlayers)+"lay_"+str(num_hidden_units)+"hu_"+str(dropout_prob)+"do_" + str(num_previous_items) + "prevItems_" + str(use_shared_embedding_model) + "shared_" + dataset

print(model_save_name)


#Set dataset params
if dataset == "movielens20m":
	data_path = "./data/movielens/"#'/data1/movielens/ml-20m'
elif dataset == "amazon_videoGames":
	data_path = "./data/amazon_videogames/data_tables_split_80_10_10"
elif dataset == "ml1m":
	data_path = "./data/ml1m/data_tables_split_85_5_10"
elif dataset == "beeradvocate":
	data_path = "./data/beeradvocate/data_tables_split_80_10_10"
elif dataset == "amazon_automotive":
	data_path = "./data/amazon_automotive/data_tables_split_80_10_10"
elif dataset == "amazon_clothing":
	data_path = "./data/amazon_clothing/data_tables_split_80_10_10"

model_save_name += "_" + dataset + "_"
modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
model_save_name += modelRunIdentifier #Append a unique identifier to the filename

print("Loading data for " + dataset)
siamese_data_reader = data_reader(data_path)

if use_shared_embedding_model:
	m = SiameseRecNet_shared(siamese_data_reader.num_users, siamese_data_reader.num_items, num_previous_items, numlayers, num_hidden_units, embedding_size, activation_type, dropout_prob)
else:
	m = SiameseRecNet(siamese_data_reader.num_users, siamese_data_reader.num_items, num_previous_items, numlayers, num_hidden_units, embedding_size, activation_type, dropout_prob)
m.cuda()
criterion = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

if optimizer_type == "adam":
	optimizer = torch.optim.Adam(m.parameters(), weight_decay=l2_regularization)
elif optimizer_type == "rmsprop":
	optimizer = torch.optim.RMSprop(m.parameters(), weight_decay=l2_regularization)
else:
	raise(notImplementedError())

min_loss = None
best_epoch = 0
train_history = []
val_history = []
best_model = None
train_gen = siamese_data_reader.data_gen(batch_size, "train", num_previous_items, subset_size=subset_size)
valid_gen = siamese_data_reader.data_gen(batch_size, "valid", num_previous_items, subset_size=subset_size)

train_epoch_length = int(math.ceil(siamese_data_reader.num_train_ratings/batch_size))
val_epoch_length = int(math.ceil(siamese_data_reader.num_valid_ratings/batch_size))

for i in range(max_epochs):
	print("\nStarting epoch ", i, " with minibatch size ", batch_size)	

	#Train model
	print("Training model")
	cumulative_loss_epoch_train = 0
	cum_mae_train = 0
	for j in range(train_epoch_length):
		inputs, targets = next(train_gen)
		preds = m(inputs) #Find a way to make prettier

		loss = criterion(preds, targets)
		mae = mae_loss(preds, targets)
		cumulative_loss_epoch_train += loss.data.cpu().numpy()[0]
		cum_mae_train += mae.data.cpu().numpy()[0]
		#print("Iteration: ", j, "      Loss: ", loss.data.numpy()[0], "      Average loss so far: ", cumulative_loss_epoch_train/(j+1), end='\033[K \r')
		print("Batch: ", j+1, "/", train_epoch_length, "      Loss: {:1.5f}".format(loss.data.cpu().numpy()[0]), "      Average loss so far: {:1.8f}".format(cumulative_loss_epoch_train/(j+1)), end='\033[K \r')

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	train_loss = cumulative_loss_epoch_train/train_epoch_length
	train_mae = cum_mae_train/train_epoch_length
	print("\nTrain loss for epoch ", i, " : ", train_loss)
	print("Train AUC for epoch ", i, " : ", 1-(train_mae/2))
	train_history.append(train_loss)

	#Validate model
	print("\nValidating model")
	cumulative_loss_epoch_valid = 0
	cum_mae_valid = 0
	for j in range(val_epoch_length):
		inputs, targets = next(valid_gen)
		preds = m(inputs)

		loss = criterion(preds, targets)
		mae = mae_loss(preds, targets)
		cumulative_loss_epoch_valid += loss.data.cpu().numpy()[0]
		cum_mae_valid += mae.data.cpu().numpy()[0]
		print("Batch: ", j+1, "/", val_epoch_length,"      Loss: {:1.5f}".format(loss.data.cpu().numpy()[0]), "      Average loss so far: {:1.8f}".format(cumulative_loss_epoch_valid/(j+1)), end='\033[K \r')
	
	#Early stopping code
	val_loss = cumulative_loss_epoch_valid/val_epoch_length
	val_mae = cum_mae_valid/val_epoch_length
	print("\nValidation loss for epoch ", i, " : ", val_loss)
	print("Validation AUC for epoch ", i, " : ", 1-(val_mae/2))
	val_history.append(val_loss)
	if early_stopping_metric == "loss":
		val_metric = val_loss
	elif early_stopping_metric == "mae":
		val_metric = val_mae

	if min_loss is None:
		min_loss = val_metric
		best_epoch = i
		best_model = copy.deepcopy(m)
	elif min_loss>val_metric:
		min_loss = val_metric
		best_epoch = i
		best_model = copy.deepcopy(m) #Want to replace with something that saves to disk...
		#m.save(model_save_path+model_save_name+"_epoch_"+str(i+1)+"_bestValidScore") #Only save if it is the best model (will save a lot of time and disk space...)
	elif i-best_epoch>patience:
		print("Stopping early at epoch ", i)
		print("Best epoch was ", best_epoch)
		print("Val history: ", val_history)
		break
	

"""
#Testing
try:
	best_m = keras.models.load_model(model_save_path+model_save_name+"_epoch_"+str(best_epoch+1)+"_bestValidScore")
	best_m.save(model_save_path+model_save_name+"_bestValidScore") #resave the best one so it can be found later
	test_epoch = best_epoch+1
except:
	print("FAILED TO LOAD BEST MODEL. TESTING WITH MOST RECENT MODEL.")
	best_m = m
	test_epoch = i+1
"""

print("Testing model from epoch: ", best_epoch)

test_gen = siamese_data_reader.data_gen(batch_size, "test", num_previous_items, subset_size=subset_size)
test_epoch_length = int(math.ceil(siamese_data_reader.num_test_ratings/batch_size))

cumulative_loss_epoch_test = 0
cum_mae_test = 0
for j in range(test_epoch_length):
	inputs, targets = next(test_gen)
	preds = best_model(inputs)

	loss = criterion(preds, targets)
	cumulative_loss_epoch_test += loss.data.cpu().numpy()[0]
	cum_mae_test += mae.data.cpu().numpy()[0]
	print("Loss: {:1.5f}".format(loss.data.cpu().numpy()[0]), "      Average loss so far: {:1.8f}".format(cumulative_loss_epoch_test/(j+1)), end='\033[K \r')

test_mae = cum_mae_test/test_epoch_length
print("\nTest loss: ", cumulative_loss_epoch_test/test_epoch_length)
print("Test AUC: ", 1-(test_mae/2))


