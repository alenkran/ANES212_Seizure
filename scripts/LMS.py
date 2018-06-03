################################################################################
# Description: Classes for linear filters

# Linear filters implemented so far:
#   - mu_LMS
#   - alpha_LMS
# *adapted from Christian Choe's and Min Cheol Kim's LMS filter for EE368
################################################################################

import numpy as np

# This class implements the good old LMS filter
# Requires the training_set to be a dictionary.
class mu_LMS_filter:
	def __init__(self, training_set, mu_fraction=0.5, num_weight=5, bias=False, causal=True, delay=0, alpha=False):
		self.training_set = training_set
		self.num_weight = num_weight
		self.mu_fraction = mu_fraction
		if not alpha:
			self.trace_R = self.est_trace_R()
			self.mu = self.mu_fraction / self.trace_R
		self.bias = bias
		self.causal = causal
		self.delay = delay
		self.alpha = alpha
		if bias:
			self.weight = np.zeros((self.num_weight+1))
		else:
			self.weight = np.zeros((self.num_weight))
	
	def reset_weight(self):
		self.weight = np.zeros(len(self.weight))

	def random_weight(self):
		self.weight = np.random.rand(len(self.weight))

	def set_weight(self, weight):
		self.weight = weight

	def get_weight(self):
		return self.weight

	def train_single_eeg(self, eeg_id, train_cycles=5, update=False, prediction_error=False):
		W = self.weight # initalize weights
		error_path = []
		value = self.training_set[eeg_id]
		input_x = value[:, 1] # Input
		output_y = value[:, 0] # Output (guess)
		num_sample = len(input_x)
		weight_path = np.zeros((train_cycles*num_sample,len(self.weight)))
		error_path = np.zeros((train_cycles*num_sample,1))
		if prediction_error:
			prediction_error = np.zeros((num_sample,1))
			prediction = True
		else:
			prediction = False
		
		# set start and end index for training, taking into account delay and causality
		if self.causal:
			end_idx = len(input_x)-1
			start_idx = np.max([self.num_weight-1, self.delay])
		else:
			end_idx = len(input_x)-1-(self.num_weight/2)
			start_idx = np.max([self.num_weight/2, self.delay])
		predicted = np.zeros(len(input_x))
		predicted.fill(np.nan)
		#end_idx = len(input_x)-1 if self.causal else len(input_x)-1-(self.num_weight/2)
		last_idx = end_idx+1
		for cycle in range(train_cycles):
			for i in range(start_idx, last_idx):
				if self.causal:
					indices = range(i-(self.num_weight-1),i+1)
				else:
					half = self.num_weight/2 #type is int, so no decimals
					indices = range(i-(self.num_weight-1-half),i+1+half)
				if self.bias:
					x_tap = np.concatenate([np.array([1]),input_x[indices]])
				else:
					x_tap = input_x[indices]
				error = output_y[i-self.delay] - np.dot(x_tap, W)
				if self.alpha:
					W = W + self.mu_fraction*error/(np.linalg.norm(x_tap)**2)*x_tap
				else:
					W = W + 2*self.mu*error*x_tap
				weight_path[cycle*num_sample + i] = W
				error_path[cycle*num_sample + i] = error

		# Calculate error
		if prediction:
			for i in range(start_idx, end_idx+1):
				if self.causal:
					indices = range(i-(self.num_weight-1),i+1)
				else:
					half = self.num_weight/2 #type is int, so no decimals
					indices = range(i-(self.num_weight-1-half),i+1+half)
				if self.bias:
					x_tap = np.concatenate([np.array([1]),input_x[indices]])
				else:
					x_tap = input_x[indices]
				if i > start_idx:
					predicted[i] = np.dot(x_tap, W)
				error = output_y[i-self.delay] - np.dot(x_tap, W)
				prediction_error[i] = error

		# Update weight
		if update:
			self.weight = W

		return weight_path, predicted, error_path, prediction_error
	
	def train(self, num_repeat = 10):
		num_key = len(self.training_set)
		weight_path = np.zeros((num_key*num_repeat,len(self.weight)))
		for k in range(num_repeat):
			print 'training iteration: {}'.format(k+1)
			for idx, (key, value) in enumerate(self.training_set.iteritems()):
				print 'idx'
				W = self.weight # initalize weights
				#value = np.array(value)
				input_x = value[:, 1]
				output_y = value[:, 0]
				
				# set start and end index for training, taking into account delay and causality
				start_idx = np.max([self.num_weight-1, self.delay])
				if self.causal:
					end_idx = len(input_x)-1
					start_idx = np.max([self.num_weight-1, self.delay])
				else:
					end_idx = len(input_x)-1-(self.num_weight/2)
					start_idx = np.max([self.num_weight/2, self.delay])
					
				#end_idx = len(input_x)-1 if self.causal else len(input_x)-1-(self.num_weight/2)
				for i in range(start_idx, end_idx+1):
					if self.causal:
						indices = range(i-(self.num_weight-1),i+1)
					else:
						half = self.num_weight/2 #type is int, so no decimals
						indices = range(i-(self.num_weight-1-half),i+1+half)
					if self.bias:
						x_tap = np.concatenate([np.array([1]),input_x[indices]])
					else:
						x_tap = input_x[indices]
					error = output_y[i-self.delay] - np.dot(x_tap,W)
					if self.alpha:
						W = W + self.mu_fraction*error/(np.linalg.norm(x_tap)**2)*x_tap
					else:
						W = W + 2*self.mu*error*x_tap
				self.weight = W # update weights after each patient
				weight_path[k*num_key + idx] = W
		return weight_path
	
	# test signal should be a 1d array
	def apply_filter(self, test_signal, output = None):
		filtered_signal = np.zeros(len(test_signal))
		filtered_signal.fill(np.nan)
		if output is not None:
			error = np.copy(filtered_signal)
		
		# set end idex for filtering, taking into causality
		if self.causal:
			end_idx = len(test_signal)-1
		else:
			end_idx = len(test_signal)-1-(self.num_weight/2)
		#end_idx = len(input_x)-1 if self.causal else len(input_x)-1-(self.num_weight/2)
		
		for i in range(self.num_weight-1, end_idx+1):
			if self.causal:
				indices = range(i-(self.num_weight-1),i+1)
			else:
				half = self.num_weight/2 #type is int, so no decimals
				indices = range(i-(self.num_weight-1-half),i+1+half)
			if self.bias:
				x_tap = np.concatenate([np.array([1]),test_signal[indices]])
			else:
				x_tap = test_signal[indices]
			filtered_signal[i] = np.dot(x_tap, self.weight)
			if output is not None:
				error[i] = output[i-self.delay] - filtered_signal[i]

		# apply delay
		filler = np.zeros(self.delay)
		filler.fill(np.nan)
		filtered_signal = np.concatenate([filtered_signal[self.delay:], filler])
		if output is not None:
			return filtered_signal, error
		else:
			return filtered_signal
	
	def est_trace_R(self):
		s_sq_avg = np.zeros((self.num_weight))
		for key, value in self.training_set.iteritems():
			y = value[:, 0]
			s_sq = np.zeros((self.num_weight))
			for i in range(len(y)-self.num_weight+1):
				x = y[i:i+self.num_weight]
				s_sq = s_sq + x*x
			s_sq = s_sq/i
			s_sq_avg = s_sq_avg + s_sq
		s_sq_avg = s_sq_avg/len(self.training_set)
		return sum(s_sq_avg)