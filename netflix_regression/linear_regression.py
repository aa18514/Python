import numpy as np
import matplotlib.pyplot as plt 
import scipy 
from sklearn import linear_model
import math
import argparse
import textwrap
from itertools import chain
import datetime 
from sklearn.cross_validation import KFold
from datetime import timedelta
from file_reader import file_reader
from joblib import Parallel, delayed
import multiprocessing


def quantize(expected_ratings): 
	ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.50]
	expected_ratings[np.where(expected_ratings < 0.25)]  = 0.00
	for val in ratings: 
		expected_ratings[np.where(np.logical_and(expected_ratings >= val - 0.25, expected_ratings <= val))] = val
		expected_ratings[np.where(np.logical_and(expected_ratings > val, expected_ratings < val + 0.25))] = val
	expected_ratings[np.where(expected_ratings >= 4.75)] = 5.00
	return expected_ratings

def compute_error(weights, test_ratings, movie_features): 
	expected_ratings = quantize(np.dot(weights, np.array(movie_features).T))
	return(np.mean((expected_ratings - test_ratings)**2))

def train_dataset(featureDimension, features, ratings, lambd_val):
	clf = linear_model.Ridge(alpha=lambd_val, normalize = False, fit_intercept = False, solver = 'lsqr')
	clf.fit(features.reshape(len(ratings), featureDimension),ratings)
	return clf.coef_

def k_fold_algorithm(movie_ratings, featureDimension, res, values_of_lambda, K):
	errors = []
	cv = KFold(len(movie_ratings), n_folds = K)
	for obj in cv: 
		error = [] 
		for value in values_of_lambda:
			weight = train_dataset(featureDimension, movie_ratings[obj[0]], res[obj[0]], value)
			error.append(compute_error(weight, res[obj[1]], movie_ratings[obj[1]]))
		errors.append(error)
	errors = np.sum(errors, axis = 0, keepdims = True)
	reg_constant = values_of_lambda[np.argmin(errors)]
	return reg_constant, train_dataset(featureDimension, movie_ratings, res, reg_constant)

def naive_linear_regression(movie_ratings, res): 	
	return np.dot(np.linalg.pinv(movie_ratings), res.T)

def compute(weight, partitioned_test_ratings, partitioned_movie_features):
	errors = np.array([])
	for users in range(671):
			e = quantize(np.dot(weight[users], partitioned_movie_features[users].T))	
			errors = np.append(errors, np.mean((e - partitioned_test_ratings[users])**2))
	return errors


def processInput(i, ratings, movie_features, algorithm, args):
	print(i)
	person = ratings[(ratings[:,0] - 1) == i]
	movie_ratings = movie_features[np.where(ratings[:,0] - 1 == i)][:,1:]
	featureDimension = len(movie_ratings[0])
	w = None
	rg_constant = None
	if(algorithm == "lin_reg"):
		w = naive_linear_regression(movie_ratings, person[:,2])
	elif(algorithm == "k_fold"):
		rg_constant, w = k_fold_algorithm(movie_ratings, featureDimension, person[:,2], args[0], args[1])
	return [rg_constant], movie_ratings, person[:,2], w, [i]

def extract_person(ratings, algorithm, movie_features, *args, **kwargs):
		print("extract person")
		print(algorithm) 
		j = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(processInput)(i, ratings, movie_features, algorithm, args) for i in range(671)))
		regularized_constants = np.array([x for _,x in sorted(zip(j[:,4], j[:,0]))])
		weight = np.array([x for _,x in sorted(zip(j[:,4],j[:,3]))])
		partitioned_ratings = np.array([x for _,x in sorted(zip(j[:,4], j[:,2]))])	
		partitioned_movie_features = np.array([x for _,x in sorted(zip(j[:,4], j[:,1]))])
		return regularized_constants, partitioned_ratings, partitioned_movie_features, weight
		
def compute_test_error(weight, test_ratings, movie_features):
	_, partitioned_test_ratings, partitioned_movie_features, _ = extract_person(test_ratings, "None", movie_features)
	return compute(weight, partitioned_test_ratings, partitioned_movie_features)

def compute_train_error(train_ratings, movie_features, algorithm, *args, **kwargs):
	regularized_constants, partitioned_train_ratings, partitioned_movie_features, weight = extract_person(train_ratings, algorithm, movie_features, args[0], args[1])
	return regularized_constants, weight, compute(weight, partitioned_train_ratings, partitioned_movie_features)

def func(k, movie_features, train_data): 
	regularized_constants, weight, train_error = compute_train_error(f.read_train_data(), movie_features, "k_fold", np.logspace(-5, 0, 100), k)
	return regularized_constants, train_error, weight

def plot_data(title, xlabel, ylabel, x, y, *args, **kwargs): 
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	if len(args) == 0: 
		plt.plot(x, y)
	else:
		plt.plot(x, y, args[0])
	plt.show()

def linear_regression_with_regularization(movie_features, train_ratings, test_ratings, args):
	"""a total of 671 users, 700003 movies.
	the function handles linear_model			
	regression for each user, 
	linear regression with regularization, 
	and non -linear transformation """
	means = []
	std = []
	if(args.verbose == 3):
		n_features = 172
	else:
		n_features = 19
	n_features += 1 
	b = np.ones((70002,n_features))
	means = np.mean(movie_features[train_ratings[:,1] - 1][:,1:], axis = 0)
	stds = np.std(movie_features[train_ratings[:,1] - 1][:,1:], axis = 0)
	b[:,2:] = (movie_features[train_ratings[:,1] - 1][:,1:] - means)/(stds + 10**-8) 
	b[:,0] = train_ratings[:,1]
	#s = f.compute_pca(features)
	
	c = np.ones((30002, n_features))	
	c[:,2:] = (movie_features[test_ratings[:,1] - 1][:,1:] - means)/(stds + 10**-8) 
	c[:,0] = test_ratings[:,1]

	
		
	if(args.verbose == 1 or args.verbose == 3): 
		K = [2, 3, 4, 5, 6, 7, 8]
		regularized_constants = []
		train_errors = []
		final_weights = []
		for i in K: 
			rg, train_error, weight = func(i, b, train_ratings)
			regularized_constants.append(rg)
			train_errors.append(train_error)
			final_weights.append(weight)
		regularized_constants = np.array(regularized_constants)
		train_errors  = np.array(train_errors)
		final_weights = np.array(final_weights)
		bias = np.array([])
		variance = np.array([])
		error = np.array([])
		users = np.arange(0, 671, 1)
		for i in range(len(K)):
			title = (' lambda vs users at K %f' % K[i])
			plot_data(title, 'users', 'values of lambda', users, regularized_constants[i], 'g*')
			bias = np.append(bias, np.mean(train_errors[i]))
			variance = np.append(variance, np.var(train_errors[i]))
		error = bias + variance
		plot_data('bias against K', 'K', 'bias train data', K, bias)
		plot_data('variance against K', 'K', 'variance train data', K, variance)
		plot_data('total error against K', 'K', 'error train data', K, error)
		print(error)
		minimum = np.argmin(error)
		return train_errors[minimum], compute_test_error(final_weights[minimum], test_ratings, c)
	else: 
		_, weight, train_error = compute_train_error(train_ratings, b, "lin_reg", None, None)
		return train_error, compute_test_error(weight, test_ratings, c)

def exponential_weightings(error, beta): 
	vo = 0.0
	error = (1 - beta) * error
	vs = []
	for i in range(len(error)):
		vo = (beta * vo + error[i])
		vs.append(vo)
	return vs

def regression_analysis(movie_features, train_ratings, test_ratings, args):
	beta = 0.9
	a = datetime.datetime.now()
	error_train, error_test = linear_regression_with_regularization(movie_features, train_ratings, test_ratings, args)
	b = datetime.datetime.now()
	plt.xlabel("users")
	plt.ylabel("exponentially weighted squared error")
	plt.plot(np.arange(0., len(error_test), 1), exponential_weightings(error_test, beta))
	plt.plot(np.arange(0., len(error_train), 1), exponential_weightings(error_train, beta))
	plt.show()
	print("program took: %f s" % ((b-a).total_seconds()))
	print("train bias: %f"  % (np.mean(error_train)))
	print("train var:  %f"  % (np.var(error_train)))
	print("test bias:  %f"  % (np.mean(error_test)))
	print("test var:   %f"  % (np.var(error_test)))


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
	f = file_reader("movie-data\\movie-features.csv", "movie-data\\ratings-train.csv", "movie-data\\ratings-test.csv")
	best_state, pearsonCoefficients, movie_features = f.read_movie_features(args)
	if(args.verbose != 0): 
		regression_analysis(movie_features, f.read_train_data(), f.read_test_data(), args)
		print("pearson coefficient between %s and %s is %f" % (best_state[0], best_state[1], best_state[2]))	
		plt.plot(pearsonCoefficients, 'g*')
		plt.xlabel('genre tuple')
		plt.ylabel('correlation coefficient')
		plt.show()