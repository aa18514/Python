import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


def ProcessInput(j):
	dataset = np.random.uniform(low = -10, high = 10, size = (j,))
	data = np.array(10**dataset, dtype = np.float32)
	sum_dataset1 = 0
	quant = []
	for i in range(j):
		if(((sum_dataset1 + data[i]) == sum_dataset1) and data[i] != 0):
			quant.append(data[i])
		sum_dataset1 += data[i]
	quant = np.array(quant)
	print(np.sum(quant))
	
	return np.sum(quant), (100 * np.sum(quant))/sum_dataset1

if __name__ == "__main__":
	errors = []
	n_features = 100000
	quant = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(ProcessInput)(i) for i in range(1, n_features)))
	plt.figure(0)
	plt.plot(quant[:,0], 'r+')
	p = np.poly1d(np.polyfit(np.arange(1, n_features, 1), quant[:,0], 4))
	x = np.arange(1, n_features, 1)
	plt.plot(x, p(x))
	plt.xlabel('n_samples')
	plt.ylabel('Absolute error due to underflow')
	plt.figure(1)
	plt.plot(quant[:,1], 'r+')
	p = np.poly1d(np.polyfit(np.arange(1, n_features, 1), quant[:,1], 4))
	plt.plot(x, p(x))
	plt.xlabel('n_samples')
	plt.ylabel('Percentage error due to underflow')
	plt.show()
	print(n_features * np.sum(quant[:,0]))