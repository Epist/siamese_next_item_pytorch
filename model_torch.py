#Model using independent embeddings for each location in which an item can show up

import torch
from torch.autograd import Variable
from torch import nn
from collections import OrderedDict
import copy
from torch.nn import functional as F
import torch.nn.init as init

class SiameseRecNet(torch.nn.Module):
	def __init__(self, num_users, num_items, num_previous_items, num_hidden_layers, num_hidden_units, embedding_size, activation_type, dropout_prob, use_masking):
		super(SiameseRecNet, self).__init__()

		self.num_items = num_items
		self.num_users = num_users
		self.num_previous_items = num_previous_items
		self.num_hidden_layers = num_hidden_layers
		self.num_hidden = num_hidden_units
		self.embedding_size = embedding_size
		self.input_dim = num_users+num_items*(num_previous_items+1)
		self.activation_type = activation_type
		self.dropout_prob = dropout_prob
		self.use_masking = use_masking

		self.next_item_embedding = torch.nn.Embedding(self.num_items+1, self.embedding_size) #All items share an embedding
		init.xavier_uniform(self.next_item_embedding.weight)
		self.prev_item_embeddings = []
		for i in range(num_previous_items):
			self.prev_item_embeddings.append(torch.nn.Embedding(self.num_items+1, self.embedding_size).cuda())
			init.xavier_uniform(self.prev_item_embeddings[i].weight)
		self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_size)
		init.xavier_uniform(self.user_embedding.weight)

		if self.embedding_size != self.num_hidden:
			raise(exception("Embedding size must be the same as num hidden units for this model!"))

		self.siamese_layers_dict = OrderedDict()
		for i in range(self.num_hidden_layers):
			if i==0: #Let first layer be set manually
				#self.siamese_layers_dict["sparse_linear_input"] = OneDSparseLinear(self.input_dim, self.num_hidden)
				#self.siamese_layers_dict["embedding_input"] = torch.nn.Embedding(self.input_dim, self.num_hidden)
				pass
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
		self.siamese_layers_dict["sigmoid_output"] = nn.Sigmoid()

		self.siamese_half = nn.Sequential(self.siamese_layers_dict)

	def forward(self, input_list):

		#left_input_indices, left_input_values, left_input_size 	= self.sparse_cat(previous_item_inputs_tensors + user_tensor + left_cand_tensor)
		#right_input_indices, right_input_values, right_input_size = self.sparse_cat(previous_item_inputs_tensors + user_tensor + right_cand_tensor)
		#print(left_inputs.size())

		left_embeddings = self.next_item_embedding(input_list[1]) #Left candidate item
		right_embeddings = self.next_item_embedding(input_list[2]) #Right candidate item

		user_input_embedding = self.user_embedding(input_list[0])
		left_embeddings += user_input_embedding
		right_embeddings += user_input_embedding

		prev_item_inputs = input_list[3:self.num_previous_items+3]
		if self.use_masking:
			prev_item_masks = input_list[self.num_previous_items+3:]
		for i, cur_prev_item_input in enumerate(prev_item_inputs): #Previous items
			#if torch.lt(cur_prev_item_input.data, torch.LongTensor(self.num_items).cuda()):
			if self.use_masking:
				cur_prev_item_input_embedding = self.prev_item_embeddings[i](cur_prev_item_input).t()
				cur_mask = prev_item_masks[i]
				masked_embedding = cur_prev_item_input_embedding.t() * cur_mask.expand_as(cur_prev_item_input_embedding).t()
				#masked_embedding = masked_embedding.t()
				left_embeddings += masked_embedding
				right_embeddings += masked_embedding
			else:
				cur_prev_item_input_embedding = self.prev_item_embeddings[i](cur_prev_item_input)
				left_embeddings += cur_prev_item_input_embedding
				right_embeddings += cur_prev_item_input_embedding


		#left_embeddings.append(self.user_embedding(left_inputs[num_previous_items])) #User
		#right_embeddings.append(self.user_embedding(right_inputs[num_previous_items]))

		left_output = self.siamese_half(left_embeddings)
		right_output = self.siamese_half(right_embeddings)

		return left_output - right_output




	def accumu(self, l):
		total = 0
		for x in l:
			total += x
			yield total

	def sparse_cat(self, l, dim=1):
		idxs = [copy.deepcopy(i.data._indices()) for i in l]
		shifts = list(self.accumu(i.size()[dim] for i in l))
		for i in range(len(idxs) - 1):
			idxs[i + 1][dim, :] += shifts[i]
		all_index = torch.cat(idxs, 1)
		all_values = torch.cat([i.data._values() for i in l])
		all_size = list(copy.deepcopy(l[0].size()))
		all_size[dim] = shifts[-1]
		print(all_size)
		return [Variable(torch.LongTensor(all_index), requires_grad=False), Variable(torch.FloatTensor(all_values), requires_grad=True), torch.Size(all_size)]


class OneDSparseLinear(nn.Linear):
	def forward(self, indices):
		linear_sum = Variable(torch.zeros(self.out_features), requires_grad=False)
		for i, input_index in enumerate(indices):
			#linear_sum += F.linear(torch.FloatTensor([1]), self.weight[:,input_index], self.bias) #Will replace with simpler implementation
			cur_weights = self.weight[:,input_index]
			cur_bias = self.bias
			linear_sum += cur_weights + cur_bias
		return linear_sum


"""
class OneDSparseLinear(nn.Linear):
	def forward(self, indices, values):
		linear_sum = Variable(torch.zeros(self.out_features), requires_grad=True)
		for i, input_index in enumerate(indices):
			input_value = torch.index_select(values, 0, torch.LongTensor([i]))
			cur_weight = torch.index_select(self.weight, 1, input_index)
			cur_bias = torch.index_select(self.bias, 1, input_index)
			linear_sum += F.linear(input_value, cur_weight, cur_bias)
		return linear_sum
"""