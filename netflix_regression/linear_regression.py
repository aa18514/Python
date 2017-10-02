import numpy as np
import matplotlib.pyplot as plt 
import scipy 
from sklearn import linear_model
import math
import argparse
import textwrap
from itertools import chain
from multiprocessing.dummy import Pool as ThreadPool
import datetime 
from datetime import timedelta

def quantize(expected_ratings): 
	expected_ratings[np.where(expected_ratings < 0.25)]  = 0.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 0.25, expected_ratings <= 0.50))] = 0.50
	expected_ratings[np.where(np.logical_and(expected_ratings > 0.50, expected_ratings  < 0.75))]  = 0.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 0.75, expected_ratings <= 1.00))] = 1.00
	expected_ratings[np.where(np.logical_and(expected_ratings > 1.00, expected_ratings < 1.25))]  = 1.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 1.25, expected_ratings <= 1.50))] = 1.50
	expected_ratings[np.where(np.logical_and(expected_ratings > 1.50, expected_ratings < 1.75))]  = 1.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 1.75, expected_ratings <= 2.00))] = 2.00
	expected_ratings[np.where(np.logical_and(expected_ratings > 2.00, expected_ratings < 2.25))]  = 2.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 2.25, expected_ratings <= 2.50))]  = 2.50
	expected_ratings[np.where(np.logical_and(expected_ratings > 2.50, expected_ratings < 2.75))]  = 2.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 2.75, expected_ratings <= 3.00))]  = 3.00
	expected_ratings[np.where(np.logical_and(expected_ratings > 3.00, expected_ratings < 3.25))]  = 3.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 3.25, expected_ratings <= 3.50))] = 3.50
	expected_ratings[np.where(np.logical_and(expected_ratings > 3.50, expected_ratings < 3.75))]  = 3.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 3.75, expected_ratings <= 4.00))] = 4.00
	expected_ratings[np.where(np.logical_and(expected_ratings > 4.00, expected_ratings < 4.25))]  = 4.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 4.00, expected_ratings < 4.25))]  = 4.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 4.25, expected_ratings <= 4.50))] = 4.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 4.50, expected_ratings <= 4.75))] = 4.50
	expected_ratings[np.where(expected_ratings >= 4.75)] = 5.00
	return expected_ratings

def compute_error(weights, test_ratings, movie_features): 
	expected_ratings = quantize(np.dot(weights, np.array(movie_features).T))
	return(np.mean((expected_ratings - test_ratings)**2))

def train_dataset(featureDimension, features, ratings, lambd_val):
	clf = linear_model.Ridge(alpha=lambd_val, normalize = False, fit_intercept = False, solver = 'svd')
	clf.fit(features.reshape(len(ratings), featureDimension),ratings)
	return clf.coef_

def k_fold_algorithm(movie_ratings, featureDimension, res, values_of_lambda, K):
	errors = []
	limits = np.array([[0] * (K + 1)])
	subset_size = int(len(movie_ratings)/K) 
	mod = len(movie_ratings) % K
	limits[:,1:(mod + 1)] = np.arange(1, mod + 1, 1) * (subset_size + 1)
	limits[:,(mod+1):len(limits[0])] = (np.arange(1, len(limits[0]) - mod, 1) * (subset_size)) + (mod * (subset_size + 1))
	for i in range(0, K):
		features_train = np.append(movie_ratings[:int(limits[:,i]):,], movie_ratings[int(limits[:,i+1]):len(movie_ratings):,])
		features_test  = movie_ratings[int(limits[:,i]) : int(limits[:,i+1])]
		ratings_train  = np.append(res[0:int(limits[:,i]):,], res[int(limits[:,i+1]):len(res):,]) 
		ratings_test   = res[int(limits[:,i]):int(limits[:,i+1])]
		error = [] 
		for value in values_of_lambda:
			weight = train_dataset(featureDimension, features_train, ratings_train, value)
			error.append(compute_error(weight, ratings_test, features_test))
		errors.append(error)
	errors = np.sum(errors, axis = 0, keepdims = True)
	return train_dataset(featureDimension, movie_ratings, res, values_of_lambda[np.argmin(errors)])

def naive_linear_regression(movie_ratings, res): 
	return np.dot(np.linalg.pinv(movie_ratings), res.T)

def compute(weight, partitioned_test_ratings, partitioned_movie_features):
	errors = np.array([])
	for users in range(671):
			e = quantize(np.dot(weight[users], partitioned_movie_features[users].T))	
			errors = np.append(errors, np.mean((e - partitioned_test_ratings[users])**2))
	return errors

def extract_person(ratings, algorithm, movie_features, *args, **kwargs):
		featureDimension = len(movie_features[0]) 
		partitioned_ratings = [] 
		partitioned_movie_features = []
		weight = []
		for i in range(671):
			person = ratings[ratings[:,0] == i + 1]
			temp_features = movie_features[person[:,1].astype(int) - 1][:,1:featureDimension]
			partitioned_ratings.append(person[:,2])
			movie_ratings = np.ndarray(shape=(len(person[:,2]), featureDimension))
			movie_ratings[:,0] = 1 
			std = np.std(temp_features, axis = 1, keepdims = True)
			std[std == 0] = 0.001
			movie_ratings[:,1:featureDimension] = (temp_features - np.mean(temp_features, axis = 1, keepdims = True))/std
			partitioned_movie_features.append(movie_ratings)
			if(algorithm == "k_fold"): 
				weight.append(k_fold_algorithm(movie_ratings, featureDimension, person[:,2], args[0], args[1]))
			elif(algorithm == "lin_reg"): 
				weight.append(naive_linear_regression(movie_ratings, person[:,2]))
		if(algorithm == "None"):
			return partitioned_ratings, partitioned_movie_features
		else: 
			return np.array(partitioned_ratings), np.array(partitioned_movie_features), np.array(weight)

def compute_test_error(weight, test_ratings, movie_features):
	partitioned_test_ratings, partitioned_movie_features = extract_person(test_ratings, "None", movie_features)
	return compute(weight, partitioned_test_ratings, partitioned_movie_features)

def compute_train_error(train_ratings, movie_features, algorithm, *args, **kwargs):
	partitioned_train_ratings, partitioned_movie_features, weight = extract_person(train_ratings, algorithm, movie_features, args[0], args[1])
	return weight, compute(weight, partitioned_train_ratings, partitioned_movie_features)

def func(k): 
	train_ratings  = read_from_file("movie-data\\ratings-train.csv", args)
	movie_features = read_from_file("movie-data\\movie-features.csv", args)
	values_of_lambda = np.logspace(-4, 0, 50)
	weight, train_error = compute_train_error(train_ratings, movie_features, "k_fold", values_of_lambda, k)
	return (np.mean(train_error), train_error, weight)

def linear_regression_with_regularization(movie_features, train_ratings, test_ratings, args):
	"""a total of 671 users, 700003 movies.
	the function handles linear
	regression for each user, 
	linear regression with regularization, 
	and non -linear transformation """
	if(args.verbose == 1 or args.verbose == 3): 
		K = [2, 3, 4]
		results = ThreadPool(4).map(func, K)
		average_errors = list(list(zip(*results))[0])
		train_errors   = list(list(zip(*results))[1])
		final_weights  = list(list(zip(*results))[2])
		plt.plot(K, average_errors)
		plt.xlabel('K')
		plt.ylabel('average test error')
		plt.title('cross validation error against K')	
		plt.show()
		minimum = np.argmin(average_errors)
		return train_errors[minimum], compute_test_error(final_weights[minimum], test_ratings, movie_features)
	else: 
		weight, train_error = compute_train_error(train_ratings, movie_features, "lin_reg", None, None)
		return train_error, compute_test_error(weight, test_ratings, movie_features)

def read_from_file(filename, args):
	data = np.genfromtxt(filename, delimiter = ',', dtype = float)
	return data[1:]

def regression_analysis(movie_features, train_ratings, test_ratings, args):
	a = datetime.datetime.now()
	error_train, error_test = linear_regression_with_regularization(movie_features, train_ratings, test_ratings, args)
	b = datetime.datetime.now()
	print((b - a).total_seconds())
	plt.xlabel("users")
	plt.ylabel("squared error")
	plt.plot(np.arange(0., len(error_test), 1), error_test)
	plt.plot(np.arange(0., len(error_train), 1), error_train)
	plt.show()
	print("train error: %f" % (np.mean(error_train)))
	print("test error: %f"  % (np.mean(error_test)))


if __name__ == "__main__": 
	parser = argparse.ArgumentParser(
			formatter_class=argparse.RawDescriptionHelpFormatter,
			description = textwrap.dedent
			('''\
					netflix dataset for 671 users and 9066 movies
					all data is stored in csv files in the sub-directory /movie-data
					all expected ratings are quantised to the nearest 0.5
					use -v to enable linear regression with cross fold validation
					use --v to enable naive linear regression
			''')

	)
	parser.add_argument('-v', '--verbose', action="count", help = "used to switch between linear regression with and w/o cross_validation")
	args = parser.parse_args()
	movie_features = read_from_file("movie-data\\movie-features.csv", args)
	test_ratings   = read_from_file("movie-data\\ratings-test.csv", args)
	train_ratings = read_from_file("movie-data\\ratings-train.csv", args)
	if(args.verbose != 0): 
		regression_analysis(movie_features, train_ratings, test_ratings, args)