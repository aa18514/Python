import numpy as np 
import datetime

class file_reader:
	def __init__(self, movie_features, train_file, test_file):
		self.__movieFeatures = movie_features
		self.__trainFile = train_file
		self.__testFile = test_file

	def calculate_correlation_coeff(self, labels, data):
		best_state = [] 
		pearsonCoefficients = []
		featureDimension = len(data[0])
		minimum = -0.0001
		x = data[:,1:featureDimension]
		x_mean = np.mean(x, keepdims = True, axis = 0)
		x2_mean = np.mean(x**2, keepdims = True, axis = 0)
		for i in range(1, featureDimension): 
			for j in range(i + 1, featureDimension):
				y = data[:,j]
				y_mean = x_mean[:,(j - 1)]
				pearsonCoefficient = np.mean(x[:,(i - 1)] * y) - x_mean[:,(i - 1)] * y_mean/np.sqrt((x2_mean[:,(i-1)] - (x_mean[:,(i-1)]**2)) * (np.mean(y**2) - (y_mean * y_mean))) 
				if(pearsonCoefficient < minimum): 
					best_state = [labels[i], labels[j], pearsonCoefficient]
					minimum = pearsonCoefficient
				pearsonCoefficients.append(pearsonCoefficient)
		return best_state, pearsonCoefficients	

	def read_movie_features(self, args): 
		data = np.genfromtxt(self.__movieFeatures, delimiter = ',', dtype = float)
		data = data[1:]
		data[:,0] = 1
		with open(self.__movieFeatures) as f: 
			labels = f.readline()
			labels = labels.split()
			featureDimension = len(data[0])
			a = datetime.datetime.now()
			best_state, pearsonCoefficients = self.calculate_correlation_coeff(labels, data)
			b = datetime.datetime.now()
			print((b-a).total_seconds())
			if(args.verbose == 3): 
				temp = np.array([[0] * 172] * len(data))
				temp[:,0:featureDimension] = data[:,0:featureDimension]
				curr = featureDimension
				for i in range(1, featureDimension): 
					x = data[:,i]
					for j in range(i+1, featureDimension):
						y = data[:,j]
						temp[:, curr] = x * y
						curr = curr + 1 
				mean = np.mean(temp[:,1:172], axis = 1, keepdims = True)
				std = np.std(temp[:,1:172], axis = 1, keepdims = True)
				std[std == 0] = 0.0001
				temp[:,1:172] = (temp[:,1:172] - mean)/std
				return best_state, pearsonCoefficients, temp
			else: 
				mean = np.mean(data[:,1:19], axis = 1, keepdims = True )
				std = np.std(data[:,1:19], axis = 1, keepdims = True)
				std[std == 0] = 0.0001
				data[:,1:19] = (data[:,1:19] - mean)/std
				return best_state, pearsonCoefficients, data
	
	def read_train_data(self): 
		data = np.genfromtxt(self.__trainFile, delimiter = ',', dtype = int)
		data = data[1:]
		return data

	def read_test_data(self): 
		data = np.genfromtxt(self.__testFile, delimiter = ',', dtype = int)
		data = data[1:]
		return data