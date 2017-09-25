import numpy as np
import matplotlib.pyplot as plt 
import scipy 
from sklearn import linear_model
from sklearn import decomposition
import math
import argparse
import textwrap

variance = -1
mean = -1

def compute_error(weights, test_ratings, movie_features): 
	expected_ratings = np.dot(weights, np.array(movie_features).T)
	expected_ratings[np.where(expected_ratings < 0.25)]  = 0.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 0.25, expected_ratings <= 0.50))]  = 0.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 0.50, expected_ratings < 0.75))]  = 0.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 0.75, expected_ratings <= 1.00))]  = 1.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 1.00, expected_ratings < 1.25))]  = 1.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 1.25, expected_ratings <= 1.50))]  = 1.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 1.50, expected_ratings < 1.75))]  = 1.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 1.75, expected_ratings <= 2.00))]  = 2.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 2.00, expected_ratings < 2.25))]  = 2.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 2.25, expected_ratings < 2.50))]  = 2.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 2.50, expected_ratings < 2.75))]  = 2.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 2.75, expected_ratings < 3.00))]  = 3.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 3.00, expected_ratings < 3.25))]  = 3.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 3.50, expected_ratings < 3.75))]  = 3.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 3.25, expected_ratings <= 3.50))]  = 3.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 4.00, expected_ratings < 4.25))]  = 4.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 3.75, expected_ratings <= 4.00))]  = 4.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 4.00, expected_ratings < 4.25))] = 4.00
	expected_ratings[np.where(np.logical_and(expected_ratings >= 4.25, expected_ratings <= 4.50))]  = 4.50
	expected_ratings[np.where(np.logical_and(expected_ratings >= 4.50, expected_ratings <= 4.75))]  = 4.50
	
	expected_ratings[np.where(expected_ratings >= 4.75)] = 5.00
	return(np.sum((expected_ratings - test_ratings)**2)/len(test_ratings))

def standardize(movie_features, dataset):
	global mean
	global variance
	standardized_movie_features = [1]*(len(movie_features)+1)
	if(np.std(movie_features[1:]) == 0): 
		standardized_movie_features[1:] = movie_features[1:] - np.mean(movie_features[1:])
		mean = np.mean(movie_features[1:])
	elif(dataset == "train"):
		standardized_movie_features[1:] = (movie_features[1:] - np.mean(movie_features[1:]))/np.std(movie_features[1:])
		mean = np.mean(movie_features[1:]) 
		variance = np.std(movie_features[1:])
	elif(dataset == "test"): 
		if(variance != 0): 
			standardized_movie_features[1:] = (movie_features[1:] - mean)/variance
		else: 
			standardized_movie_features[1:] = (movie_features[1:] - mean)
	return standardized_movie_features

def train_dataset(features, ratings, lambd_val):
	clf = linear_model.Ridge(alpha=lambd_val, normalize = False, fit_intercept = False, solver = 'svd')
	clf.fit(features, ratings)
	return clf.coef_

def k_fold_algorithm(movie_ratings, res, values_of_lambda, K):
	errors = [] 
	for value in values_of_lambda:
		error = 0
		limit = 0
		limits = [0]*(K+1)
		subset_size = int(len(movie_ratings)/K) 
		mod = len(movie_ratings) % K
		for i in range(0, mod):
			limit += subset_size + 1 
			limits[i+1] = limit
		for i in range(mod, len(limits) -1):
			limit += subset_size
			limits[i+1] = limit	
		for i in range(0, K): 
			features_train = movie_ratings[:limits[i]] + movie_ratings[limits[i+1]:]
			features_test = movie_ratings[limits[i]:limits[i+1]]
			ratings_train = res[:limits[i]] + res[limits[i+1]:]
			ratings_test = res[limits[i]:limits[i+1]]
			weight = train_dataset(features_train, np.array(ratings_train)[:,2], value)
			error += compute_error(weight, np.array(ratings_test)[:,2], features_test)
		errors.append(error)
	weight = train_dataset(movie_ratings, np.array(res)[:,2], values_of_lambda[np.argmin(errors)])
	return weight

def naive_linear_regression(movie_ratings, res): 
		X_ = np.linalg.pinv(np.array(movie_ratings))
		return np.dot(X_, res.T)

def retrieve_sub_dataset(ratings, current_pointer, sub_data_set_label, movie_features): 
	movie_ratings = []
	res = [ratings[current_pointer]]
	curr = current_pointer - 1
	for o in range(curr + 1, len(ratings)): 
		if(ratings[o][0] == ratings[curr][0]):
			res.append(ratings[o])
			curr = curr + 1 
		else: 
			break;
	for i in range(0, len(res)): 
			movie_ratings.append(standardize((movie_features[int(res[i][1] - 1.0)]), sub_data_set_label))
	return res, movie_ratings, curr+2

def compute_test_error(test_ratings, movie_features, weight):
	current_user = 0
	j = 1 
	test_error = []
	while(j < len(test_ratings)):
		test_res, test_movie_ratings, j = retrieve_sub_dataset(test_ratings, j, "test", movie_features)
		test_error.append(compute_error(weight[current_user], np.array(test_res)[:,2], test_movie_ratings))
		current_user = current_user + 1
	return test_error

def compute_train_error(train_ratings, movie_features, algorithm, *args, **kwargs): 
	j = 1
	train_error = []
	weight = [] 
	while(j < len(train_ratings)): 
		train_res, train_movie_ratings, j = retrieve_sub_dataset(train_ratings, j, "train", movie_features)
		if(algorithm == "k_fold"): 
			weight.append(k_fold_algorithm(train_movie_ratings, train_res, args[0], args[1]))
		else:
			weight.append(naive_linear_regression(train_movie_ratings, np.array(train_res)[:,2]))
		train_error.append(compute_error(weight[-1], np.array(train_res)[:,2], train_movie_ratings))
	return weight, train_error

def linear_regression_with_regularization(movie_features, train_ratings, values_of_lambda, test_ratings, args):
	"""a total of 671 users, 700003 movies.
	the function handles linear
	regression for each user, 
	linear regression with regularization, 
	and non -linear transformation """
	if(args.verbose == 1): 
		average_errors = []
		K = [2, 3, 4, 5, 6]
		train_errors = []
		final_weights = []
		for k_val in K: 
			weight, train_error = compute_train_error(train_ratings, movie_features, "k_fold", values_of_lambda, k_val)
			final_weights.append(weight)
			train_errors.append(train_error)
			average_errors.append(np.mean(train_error))
		plt.plot(K, average_errors)
		plt.xlabel('K')
		plt.ylabel('average test error')
		plt.title('cross validation error against K')	
		plt.show()
		minimum = np.argmin(average_errors)
		return final_weights[minimum], train_errors[minimum], compute_test_error(test_ratings, movie_features, weight)
	else: 
		weight, train_error = compute_train_error(train_ratings, movie_features, "lin_reg")
		return weight, train_error, compute_test_error(test_ratings, movie_features, weight)

def read_from_file(filename):
	res = []  
	with open(filename, "r") as csvreader: 
		csvreader.readline()
		result = csvreader.readlines()
		for i in range(0, len(result)): 
			line = result[i].strip('\n')
			temp = line.split(",")
			res.append([float(i) for i in temp])
		return res

def regression_analysis(regularization_constants, movie_features, train_ratings, test_ratings, args):
	weights, error, error_test = linear_regression_with_regularization(movie_features, train_ratings, regularization_constants, test_ratings, args)
	plt.xlabel("users")
	plt.ylabel("squared error")
	t = np.arange(0., len(error_test), 1) 
	plt.plot(t, error_test)
	t = np.arange(0., len(error), 1)
	plt.plot(t, error)
	plt.show()
	print("train error: %f" % (np.mean(error)))
	print("test error: %f" % (np.mean(error_test)))


if __name__ == "__main__": 
	parser = argparse.ArgumentParser(
			formatter_class=argparse.RawDescriptionHelpFormatter,
			description = textwrap.dedent('''\
											netflix dataset for 671 users and 9066 movies
											all data is stored in csv files in the sub-directory /movie-data
										 	all expected ratings are quantised to the nearest 0.5
										 	use -v to enable linear regression with cross fold validation
										 	use --v to enable naive linear regression
										 ''')

	)
	parser.add_argument('-v', '--verbose', action="count", help = "used to switch between linear regression with and w/o cross_validation")
	args = parser.parse_args()
	movie_features = read_from_file("movie-data\\movie-features.csv")
	test_ratings   = read_from_file("movie-data\\ratings-test.csv")
	train_ratings  = read_from_file("movie-data\\ratings-train.csv")
	regularization_constants = np.logspace(-4, 0, 50)
	if(args.verbose == 1 or args.verbose == 2): 
		regression_analysis(regularization_constants, movie_features, train_ratings, test_ratings, args)