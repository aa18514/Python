import numpy as np 
import datetime

class file_reader:
	data = {}
	def __init__(self, movie_features, train_file, test_file):
		self.__movieFeatures = movie_features
		self.__trainFile = train_file
		self.__testFile = test_file
		self.data['movies'] = np.genfromtxt(self.__movieFeatures, delimiter = ',', dtype = float)[1:]
		self.data['test']   = np.genfromtxt(self.__testFile, delimiter = ',', dtype = int)[1:]
		self.data['train']  = np.genfromtxt(self.__trainFile, delimiter = ',', dtype = int)[1:]
		self.data['movies'][:,0] = 1 

	def calculate_correlation_coeff(self, labels):
		data = self.data['movies']
		best_state = [] 
		pearsonCoefficients = []
		featureDimension = len(data[0])
		minimum = -0.0001
		x = data[:,1:featureDimension]
		x_mean = np.mean(x, keepdims = True, axis = 0)
		x2_mean = np.mean(x**2, keepdims = True, axis = 0)
		for i in range(1, featureDimension): 
			y = data[:,(i+1):featureDimension]
			y_mean = x_mean[:,i:featureDimension - 1]
			for j in range(i, featureDimension - 1):
				y_mean = x_mean[:,j]
				pearsonCoefficient = np.mean(x[:,(i - 1)] * y[:,(j - i)]) - x_mean[:,(i - 1)] * y_mean/np.sqrt((x2_mean[:,(i-1)] - (x_mean[:,(i-1)]**2)) * (np.mean(y[:,(j - i)]**2) - (y_mean * y_mean))) 
				if(pearsonCoefficient < minimum): 
					best_state = [labels[i], labels[j+1], pearsonCoefficient]
					minimum = pearsonCoefficient
				pearsonCoefficients.append(pearsonCoefficient)
		return best_state, pearsonCoefficients	

	def non_linear_transformation(self):
		featureDimension = len(self.data['movies'][0])
		temp = np.array([[0] * 172] * len(self.data['movies']))
		temp[:,0:featureDimension] = self.data['movies'][:,0:featureDimension]
		curr = featureDimension
		for i in range(1, featureDimension): 
			x = self.data['movies'][:,i]
			for j in range(i+1, featureDimension): 
				y = self.data['movies'][:,j]
				temp[:,curr] = x * y
				curr = curr + 1 
			mean = np.mean(temp[:,1:172], axis = 1, keepdims = True)
			std = np.std(temp[:,1:172], axis = 1, keepdims = True)
			std[std == 0] = 0.0001
			temp[:,1:172] = (temp[:,1:172] - mean)/std
		return temp 

	def read_movie_features(self, args): 
		with open(self.__movieFeatures) as f: 
			labels = f.readline()
			labels = labels.split()
			featureDimension = len(self.data['movies'][0])
			a = datetime.datetime.now()
			best_state, pearsonCoefficients = self.calculate_correlation_coeff(labels)
			b = datetime.datetime.now()
			print((b-a).total_seconds())
			if(args.verbose == 3): 
				self.data['movies'] = self.non_linear_transformation()
			else: 
				mean = np.mean(self.data['movies'][:,1:19], axis = 1, keepdims = True )
				std = np.std(self.data['movies'][:,1:19], axis = 1, keepdims = True)
				std[std == 0] = 0.0001
				self.data['movies'][:,1:19] = (self.data['movies'][:,1:19] - mean)/std
			return best_state, pearsonCoefficients, self.data['movies']
	
	def read_train_data(self):
		return self.data['train']

	def read_test_data(self): 
		return self.data['test']