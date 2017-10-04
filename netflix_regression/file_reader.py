import numpy as np 

class file_reader:
	def __init__(self, movie_features, train_file, test_file):
		self.__movieFeatures = movie_features
		self.__trainFile = train_file
		self.__testFile = test_file

	def read_movie_features(self, args): 
		data = np.genfromtxt(self.__movieFeatures, delimiter = ',', dtype = float)
		data = data[1:]
		data[:,0] = 1
		best_state = [] 
		pearsonCoefficients = []
		with open(self.__movieFeatures) as f: 
			labels = f.readline()
			labels = labels.split()
			minimum = -0.0001
			featureDimension = len(data[0])
			for i in range(1, featureDimension): 
				x = data[:,i]
				x_mean = np.mean(x)
				x2_mean = np.mean(x*x)
				for j in range(i + 1, featureDimension): 
					y = data[:,j]
					y_mean = np.mean(y)
					pearsonCoefficient = np.mean(x * y) - x_mean * y_mean/np.sqrt((x2_mean - x_mean * x_mean) * (np.mean(y * y) - (y_mean * y_mean))) 
					if(pearsonCoefficient < minimum):
						best_state = [labels[i], labels[j], pearsonCoefficient]
						minimum = pearsonCoefficient
					pearsonCoefficients.append(pearsonCoefficient)
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
				data[:,0] = 1 
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