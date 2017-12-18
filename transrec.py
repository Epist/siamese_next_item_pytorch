import torch
from torch.autograd import Variable
from torch import nn
from collections import OrderedDict
import copy
from torch.nn import functional as F
import torch.nn.init as init

class SiameseRecNet(torch.nn.Module):
	def __init__(self, num_users, num_items, num_previous_items, num_hidden_layers, num_hidden_units, embedding_size, activation_type, dropout_prob, use_masking, use_gpu):
		super(SiameseRecNet, self).__init__()

		self.num_items = num_items
		self.num_users = num_users
		self.num_previous_items = num_previous_items
		self.num_hidden_layers = num_hidden_layers
		self.num_hidden = num_hidden_units
		self.input_dim = num_users+num_items*(num_previous_items+1)
		self.activation_type = activation_type
		self.embedding_size = embedding_size
		self.dropout_prob = dropout_prob
		self.use_masking = use_masking
		self.use_gpu = use_gpu

		self.item_embedding = torch.nn.Embedding(self.num_items+1, self.embedding_size) #All items share an embedding
		self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_size)
		#init.constant(self.item_embedding.weight, 0)
		#init.constant(self.user_embedding.weight, 0)
		init.xavier_uniform(self.item_embedding.weight)
		init.xavier_uniform(self.user_embedding.weight)

		#self.user_bias = Variable(torch.zeros(self.embedding_size), requires_grad=True)
		#init.xavier_uniform(self.user_bias)

		self.user_input_layer = nn.Linear(self.embedding_size, self.embedding_size)
		init.xavier_uniform(self.user_input_layer.weight)
		self.next_item_input_layer = nn.Linear(self.embedding_size, self.embedding_size)
		init.xavier_uniform(self.next_item_input_layer.weight)
		self.prev_items_hidden_layers = []
		for i in range(num_previous_items):
			if self.use_gpu:
				hidden = nn.Linear(self.embedding_size, self.embedding_size).cuda()
			else:
				hidden = nn.Linear(self.embedding_size, self.embedding_size)
			self.prev_items_hidden_layers.append(hidden)
			init.xavier_uniform(hidden.weight)
		
		if activation_type == "tanh":
			self.post_embedding_nonlinearity = nn.Tanh()
		elif activation_type == "sigmoid":
			self.post_embedding_nonlinearity = nn.Sigmoid()
		elif activation_type == "relu":
			self.post_embedding_nonlinearity = nn.ReLU()

		self.siamese_layers_dict = OrderedDict()
		for i in range(self.num_hidden_layers):
			if i==0: #Let first layer be set manually
				merge_size = self.embedding_size*(self.num_previous_items+2)
				#print("input merge_size", merge_size)
				self.siamese_layers_dict["linear_"+str(i)] = nn.Linear(merge_size, self.num_hidden)
				init.xavier_uniform(self.siamese_layers_dict["linear_"+str(i)].weight)
			else:
				self.siamese_layers_dict["linear_"+str(i)] = nn.Linear(self.num_hidden, self.num_hidden)
				init.xavier_uniform(self.siamese_layers_dict["linear_"+str(i)].weight)

			if activation_type == "tanh":
				self.siamese_layers_dict["tanh_"+str(i)] = nn.Tanh()
			elif activation_type == "sigmoid":
				self.siamese_layers_dict["sigmoid_"+str(i)] = nn.Sigmoid()
			elif activation_type == "relu":
				self.siamese_layers_dict["relu_"+str(i)] = nn.ReLU()
			else:
				raise(exception("Layer type " + activation_type + " not yet implemented"))
			if self.dropout_prob != 0:
				self.siamese_layers_dict["dropout_"+str(i)] = torch.nn.modules.Dropout(p=self.dropout_prob)

		self.siamese_layers_dict["linear_output"] = nn.Linear(self.num_hidden, 1)
		init.xavier_uniform(self.siamese_layers_dict["linear_output"].weight)
		#self.siamese_layers_dict["sigmoid_output"] = nn.Sigmoid()

		self.siamese_half = nn.Sequential(self.siamese_layers_dict)

		self.output_sigmoid = nn.Sigmoid()

		self.DO = torch.nn.modules.Dropout(p=self.dropout_prob)

		self.dist = nn.modules.distance.PairwiseDistance(p=2)

	def forward(self, input_list):


		left_embedding = self.item_embedding(input_list[1]) #Left candidate item
		right_embedding = self.item_embedding(input_list[2]) #Right candidate item

		#input_size = list(input_list[0].size())[0]
		user_input_embedding = self.user_embedding(input_list[0])# + self.user_bias

		prev_item_inputs = input_list[3:self.num_previous_items+3]
		if self.use_masking:
			raise(Exception("Masking not supported"))
		for i, cur_prev_item_input in enumerate(prev_item_inputs): #Previous items
			cur_prev_item_input_embedding = self.item_embedding(cur_prev_item_input)
			post_embedding = cur_prev_item_input_embedding
			translated_embedding = post_embedding + user_input_embedding



		left_output = self.dist(left_embedding, translated_embedding)
		right_output = self.dist(right_embedding, translated_embedding)

		return self.output_sigmoid(left_output - right_output)
		"""
		if self.training:
			out = left_output - right_output
			return out
		else:
			return self.output_sigmoid(left_output - right_output)"""

	def train(self):
		super(SiameseRecNet, self).train(mode=True)
		self.siamese_half.train(mode=True)
		if self.dropout_prob != 0:
			self.DO.train(mode=True)
		for i in range(self.num_hidden_layers):
			if self.dropout_prob != 0:
				self.siamese_layers_dict["dropout_"+str(i)].train(mode=True)

	def eval(self):
		super(SiameseRecNet, self).train(mode=False)
		self.siamese_half.eval()
		if self.dropout_prob != 0:
			self.DO.eval()
		for i in range(self.num_hidden_layers):
			if self.dropout_prob != 0:
				self.siamese_layers_dict["dropout_"+str(i)].eval()

	def max_norm_clip(self, w):
		#print(torch.max(torch.norm(w, 2, 0)).expand_as(w))
		return w/(torch.max(torch.norm(w, 2, 0)).expand_as(w))
		#return w.div_(torch.max(torch.norm(w, 2, 0)).expand_as(w))

		#batch_size = inp.size()[0]
		#max = torch.max(inp, dim=1)[0].view(64,1)
		#print(max)
		#print(max.size())
		#print(inp.size())
		#out = inp/max
		#return out

	def clip_weights(self):

		self.max_norm_clip(self.item_embedding.weight)
		self.max_norm_clip(self.user_embedding.weight)
