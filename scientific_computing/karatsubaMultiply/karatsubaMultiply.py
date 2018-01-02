import math
import matplotlib.pyplot as plt
import numpy as np
import datetime
import decimal
import random
from decimal import *
import multiprocessing
from joblib import Parallel, delayed

def Karat(x1, x2):
	if(abs(x1) < 10 or abs(x2) < 10): 
		return x1 * x2
	n1 = len(str(abs(x1)))
	n2 = len(str(abs(x2)))
	n_max = int(max(n1, n2) * 0.5)
	nlog1 = math.floor(math.log(x1, 2))
	nlog2 = math.floor(math.log(x2, 2))
	a, b, c, d = 0, 0, 0, 0
	normalization_constant = 10**n_max
	with localcontext() as ctx:
		ctx.prec = max(nlog1, nlog2)
		ctx.rounding=decimal.ROUND_CEILING	
		if(x1 != x2):
			a = (x1/normalization_constant).to_integral_exact() 
			b = x1 % normalization_constant
			c = (x2/normalization_constant).to_integral_exact()
			d = x2 % normalization_constant
		else: 
			a = (x1/normalization_constant).to_integral_exact()
			b = x1 % normalization_constant
			c = a 
			d = b
	ad_plus_bc = Karat(a + b, c + d)
	b_d = Karat(b, d)
	a_c  = Karat(a, c)
	return (10**(2*n_max)) * a_c + (10**n_max) * (ad_plus_bc - a_c - b_d) + b_d


def KaratsubaMultiplyWithSpatialLocality_16(x1, x2, dictionary): 
	if(x1 == 0 or x2 == 0): 
		return 0
	elif(abs(x1) < 10 or abs(x2) < 10): 
		return x1 * x2; 
	n1 = len(str(abs(x1)))
	n2 = len(str(abs(x2)))
	n_max = int(max(n1, n2) * 0.5)
	a, b, c, d = 0, 0, 0, 0 

	normalization_constant = 10**n_max
	with localcontext() as ctx: 
		ctx.prec = 3000
		ctx.rounding = decimal.ROUND_CEILING
		a = (x1/normalization_constant).to_integral_exact() 
		b = x1 % normalization_constant
		if(x1 != x2): 
			c = (x2/normalization_constant).to_integral_exact()
			d = x2 % normalization_constant
		else: 
			c = a
			d = b
		
		sums = [a + b, c + d, a + b + c + d, b, d, c, a]
		sum_ab = sums[0]
		sum_cd = sums[1]
		sums = 16 * sums
		str_abcd = str(sum_ab) + str(sum_cd)
		str_bd = str(b) + str(d)
		str_ac = str(a) + str(c)
		if (str_abcd in dictionary.keys()): 
			pass
		else:
			dictionary[str_abcd] = KaratsubaMultiplyWithSpatialLocality_16(sum_ab, sum_cd, dictionary)
		
			for i in range(1, 16):
				dictionary[str(sum_ab + i) + str(sum_cd + i)] = dictionary[str_abcd] + i * i + sums[2]
				dictionary[str(sum_ab - i) + str(sum_cd - i)] = 2 * dictionary[str_abcd] - dictionary[str(sum_ab + i) + str(sum_cd + i)]
				dictionary[str(sum_ab) + str(sum_cd + i)] = dictionary[str_abcd] + sums[0]
				dictionary[str(sum_ab) + str(sum_cd - i)] = dictionary[str_abcd] - sums[0]
				dictionary[str(sum_ab + i) + str(sum_cd)] = dictionary[str_abcd] + sums[1]
				dictionary[str(sum_ab - i) + str(sum_cd)] = dictionary[str_abcd] - sums[1]

		if(str_bd in dictionary.keys()): 
			pass 
		else: 
			dictionary[str_bd] = KaratsubaMultiplyWithSpatialLocality_16(b, d, dictionary)
			for i in range(1, 16): 
				dictionary[(b + i) * d] = dictionary[str_bd] + sums[4]
				dictionary[(b - i) * d] = dictionary[str_bd] - sums[4]
				dictionary[b * (d + i)] = dictionary[str_bd] + sums[3]
				dictionary[b * (d - i)] = dictionary[str_bd] - sums[3]
		if(str_ac in dictionary.keys()): 
			pass
		else: 
			dictionary[str_ac] = KaratsubaMultiplyWithSpatialLocality_16(a, c, dictionary)
			for i in range(1, 16): 
				dictionary[(a + i) * c] = dictionary[str_ac] + sums[5]
				dictionary[(a - i) * c] = dictionary[str_ac] - sums[5]
				dictionary[a * (c + i)] = dictionary[str_ac] + sums[6]
				dictionary[a * (c - i)] = dictionary[str_ac] - sums[6]
	return (10**(2*n_max)) * dictionary[str_ac] + 10**n_max * (dictionary[str_abcd] - dictionary[str_ac] - dictionary[str_bd]) + dictionary[str_bd]
	

def KaratasubaMultiplywithSpatialLocality_1(x1, x2, dictionary): 
	if(x1 == 0 or x2 == 0): 
		return 0
	if(abs(x1) < 10 or abs(x2) < 10): 
		return x1 * x2
	n1 = len(str(abs(x1)))
	n2 = len(str(abs(x2)))
	n_max = int(max(n1, n2) * 0.5)
	a = 0
	b = 0
	c = 0
	d = 0
	normalization_constant = 10**n_max
	with localcontext() as ctx:
		ctx.prec = 3000
		ctx.rounding=decimal.ROUND_CEILING	
		if(x1 != x2):
			a = (x1/normalization_constant).to_integral_exact() 
			b = x1 % normalization_constant
			c = (x2/normalization_constant).to_integral_exact()
			d = x2 % normalization_constant
		else: 
			a = (x1/normalization_constant).to_integral_exact()
			b = x1 % normalization_constant
			c = a 
			d = b
	sum_ab = a + b
	sum_cd = c + d
	str_abcd = str(sum_ab) + str(sum_cd)
	str_abcd1 = str(sum_ab + 1) + str(sum_cd + 1)
	str_abcd_1 = str(sum_ab - 1) + str(sum_cd - 1)
	str_bd = str(b) + str(d)
	str_bd1 = str(b + 1) + str(d + 1)
	str_bd_1 = str(b - 1) + str(d - 1)
	str_ac = str(a) + str(c)
	str_ac1 = str(a + 1) + str(c + 1)
	str_ac_1 = str(a - 1) + str(c - 1)

	""" (a + b) * (c + d + 1) => (a + b) * (c + d) + (a + b)"""
	if(str_abcd in dictionary): 
		pass
	else:
		dictionary[str_abcd] = KaratsubaMultiplyWithSpatialLocality_1(sum_ab, sum_cd, dictionary)
		dictionary[str_abcd1] = dictionary[str_abcd] + sum_ab + sum_cd + 1
		dictionary[str_abcd_1] = dictionary[str_abcd] - sum_ab - sum_cd - 1
		dictionary[str(sum_ab) + str(sum_cd + 1)] = dictionary[str_abcd] + sum_ab
		dictionary[str(sum_ab) + str(sum_cd - 1)] = dictionary[str_abcd] - sum_ab
		dictionary[str(sum_ab + 1) + str(sum_cd)] = dictionary[str_abcd] + sum_cd
		dictionary[str(sum_ab - 1) + str(sum_cd)] = dictionary[str_abcd] - sum_cd
	if(str_bd in dictionary): 
		pass
	else: 
		sum_bd = b + d
		dictionary[str_bd] = KaratsubaMultiplyWithSpatialLocality_1(b, d, dictionary)
		dictionary[str_bd1] = dictionary[str_bd] + sum_bd + 1
		dictionary[str_bd_1] = dictionary[str_bd] - sum_bd - 1
		dictionary[str(b) + str(d + 1)] = dictionary[str_bd] + b
		dictionary[str(b) + str(d - 1)] = dictionary[str_bd] - b
		dictionary[str(b + 1) + str(d)] = dictionary[str_bd] + d
		dictionary[str(b - 1) + str(d)] = dictionary[str_bd] - d
	if(str_ac in dictionary): 
		pass
	else: 
		dictionary[str_ac] = KaratsubaMultiplyWithSpatialLocality_1(a, c, dictionary)
		sum_ac = a + c
		dictionary[str_ac1] = dictionary[str_ac] + sum_ac + 1
		dictionary[str_ac_1] = dictionary[str_ac] - sum_ac - 1
		dictionary[str(a) + str(c + 1)] = dictionary[str_ac] + a
		dictionary[str(a) + str(c - 1)] = dictionary[str_ac] - a
		dictionary[str(a + 1) + str(c)] = dictionary[str_ac] + c
		dictionary[str(a - 1) + str(c)] = dictionary[str_ac] - c
	return (10**(2*n_max)) * dictionary[str_ac] + 10**n_max * (dictionary[str_abcd] - dictionary[str_ac] - dictionary[str_bd]) + dictionary[str_bd]
	
def KaratasubaMultiplywithSpatialLocality_0(x1, x2, dictionary): 
	if(x1 == 0 or x2 == 0): 
		return 0
	if(abs(x1) < 10 or abs(x2) < 10): 
		return x1 * x2
	n1 = len(str(abs(x1)))
	n2 = len(str(abs(x2)))
	n_max = int(max(n1, n2) * 0.5)
	a, b, c, d = 0, 0, 0, 0
	normalization_constant = 10**n_max
	with localcontext() as ctx:
		ctx.prec = 3000
		ctx.rounding=decimal.ROUND_CEILING	
		if(x1 != x2):
			a = (x1/normalization_constant).to_integral_exact() 
			b = x1 % normalization_constant
			c = (x2/normalization_constant).to_integral_exact()
			d = x2 % normalization_constant
		else: 
			a = (x1/normalization_constant).to_integral_exact()
			b = x1 % normalization_constant
			c = a 
			d = b
	sum_ab = a + b
	sum_cd = c + d
	str_abcd = str(sum_ab) + str(sum_cd)
	str_abcd1 = str(sum_ab + 1) + str(sum_cd + 1)
	str_bd = str(b) + str(d)
	str_bd1 = str(b + 1) + str(d + 1)
	str_ac = str(a) + str(c)
	str_ac1 = str(a + 1) + str(c + 1)

	""" (a + b) * (c + d + 1) => (a + b) * (c + d) + (a + b)"""
	if(str_abcd in dictionary): 
		pass
	else:
		dictionary[str_abcd] = KaratsubaMultiplyWithOptimization_0(sum_ab, sum_cd, dictionary)
		dictionary[str_abcd1] = dictionary[str_abcd] + sum_ab + sum_cd + 1
		dictionary[str(sum_ab) + str(sum_cd + 1)] = dictionary[str_abcd] + sum_ab
		dictionary[str(sum_ab + 1) + str(sum_cd)] = dictionary[str_abcd] + sum_cd
	if(str_bd in dictionary): 
		pass
	else: 
		dictionary[str_bd] = KaratsubaMultiplyWithOptimization_0(b, d, dictionary)
		dictionary[str_bd1] = dictionary[str_bd] + b + d + 1
		dictionary[str(b) + str(d + 1)] = dictionary[str_bd] + b
		dictionary[str(b + 1) + str(d)] = dictionary[str_bd] + d
	if(str_ac in dictionary): 
		pass
	else: 
		dictionary[str_ac] = KaratsubaMultiplyWithOptimization_0(a, c, dictionary)
		dictionary[str_ac1] = dictionary[str_ac] + a + c + 1
		dictionary[str(a) + str(c + 1)] = dictionary[str_ac] + a
		dictionary[str(a + 1) + str(c)] = dictionary[str_ac] + c
	return (10**(2*n_max)) * dictionary[str_ac] + 10**n_max * (dictionary[str_abcd] - dictionary[str_ac] - dictionary[str_bd]) + dictionary[str_bd]
	
def KaratsubaMultiply(x1, x2): 
	if(len(str(int(x1))) == 1 or len(str(int(x2))) == 1): 
		return x1 * x2
	n1 = int(len(str(abs(x1))))
	n2 = int(len(str(abs(x2))))
	n_max = int(max(n1, n2) * 0.5)
	a, b, c, d = 0, 0, 0, 0
	with localcontext() as ctx:
		ctx.prec = 1500 # desired precision
		a = (x1/10**n_max).to_integral_exact() 
		b = x1 % 10**n_max
		c = (x2/10**n_max).to_integral_exact()
		d = x2 % 10**n_max
		b_c = KaratsubaMultiply(b, c)
	b_d = KaratsubaMultiply(b, d)
	a_c  = KaratsubaMultiply(a, c)
	a_d = KaratsubaMultiply(a, d)
	return (10**(2*n_max)) * a_c + (10**n_max) * (a_d + b_c) + b_d
	
def KaratsubaMultiplyWithOptimization(x1, x2, dictionary):
	""" 2-way set associative cache used to perform n x n multiplication """
	if(x1 == 0 or x2 == 0): 
		return 0
	if(abs(x1) < 10 or abs(x2) < 10): 
		return x1 * x2
	n1 = len(str(abs(x1)))
	n2 = len(str(abs(x2)))
	n_max = int(max(n1, n2) * 0.5)
	normalization_constant = 10**n_max
	with localcontext() as ctx:
		ctx.prec = 3000
		ctx.rounding=decimal.ROUND_CEILING	
		a = (x1/normalization_constant).to_integral_exact()
		b = x1% normalization_constant
		if(x1 != x2):
			c = (x2/normalization_constant).to_integral_exact()
			d = x2 % normalization_constant
		else: 
			c = a 
			d = b
	str_abcd = str(a + b) + str(c + d)
	str_bd = str(b) + str(d)
	str_ac = str(a) + str(c)
	if(str_abcd in dictionary): 
		pass
	else:
		dictionary[str_abcd] = KaratsubaMultiplyWithOptimization(a + b, c + d, dictionary)
	if(str_bd in dictionary): 
		pass
	else: 
		dictionary[str_bd] = KaratsubaMultiplyWithOptimization(b, d, dictionary)
	if(str_ac in dictionary): 
		pass
	else: 
		dictionary[str_ac] = KaratsubaMultiplyWithOptimization(a, c, dictionary)
	return (10**(2*n_max)) * dictionary[str_ac] + (10**n_max) * (dictionary[str_abcd] - dictionary[str_ac] - dictionary[str_bd]) + dictionary[str_bd]
	
def KaratsubaMultiply(x1, x2): 
	if(len(str(int(x1))) == 1 or len(str(int(x2))) == 1): 
		return x1 * x2
	n1 = int(len(str(abs(x1))))
	n2 = int(len(str(abs(x2))))
	n_max = int(max(n1, n2) * 0.5)
	nlog1 = math.floor(math.log(x1, 2))
	nlog2 = math.floor(math.log(x2, 2))
	a, b, c, d = 0, 0, 0, 0
	normalization_constant = 10**n_max
	with localcontext() as ctx:
		ctx.prec = max(nlog1, nlog2) # desired precision
		a = (x1/normalization_constant).to_integral_exact() 
		b = x1 % normalization_constant
		c = (x2/normalization_constant).to_integral_exact()
		d = (x2 % normalization_constant)
	b_c = KaratsubaMultiply(b, c)
	b_d = KaratsubaMultiply(b, d)
	a_c  = KaratsubaMultiply(a, c)
	a_d = KaratsubaMultiply(a, d)
	return (10**(2*n_max)) * a_c + (10**n_max) * (a_d + b_c) + b_d
	
def ProcessInput(x1, x2, experiment_numbers):
	time = 1.0
	for i in range(experiment_numbers): 
		dictionary = {}
		c = datetime.datetime.now()
		result = KaratsubaMultiplyWithOptimization(x1, x2, dictionary)
		s = datetime.datetime.now() 
		time *= (s - c).total_seconds()
	return time**(1/experiment_numbers)

def ProcessInput2(x1, x2, experiment_numbers): 
	time = 1.0 
	for i in range(experiment_numbers): 
		c = datetime.datetime.now()
		result = KaratsubaMultiply(x1, x2)
		s = datetime.datetime.now()
		time *= (s - c).total_seconds()
	return time/experiment_numbers

def ProcessInput3(x1, x2, experiment_numbers): 
	time = 1.0 
	for i in range(experiment_numbers): 
		c = datetime.datetime.now()
		result = Karat(x1, x2)
		s = datetime.datetime.now()
		time *= (s - c).total_seconds()
	return time/experiment_numbers

def ProcessInput4(x1, x2, experiment_numbers): 
	time = 1.0
	for i in range(experiment_numbers): 
		dictionary = {}
		c = datetime.datetime.now()
		result = KaratasubaMultiplywithSpatialLocality_0(x1, x2, dictionary)
		s = datetime.datetime.now()
		time *= (s - c).total_seconds()
	return time**(1/experiment_numbers)

def ProcessInput5(x1, x2, experiment_numbers): 
	time = 1.0
	for i in range(experiment_numbers): 
		dictionary = {}
		c = datetime.datetime.now()
		result = KaratasubaMultiplywithSpatialLocality_1(x1, x2, dictionary)
		s = datetime.datetime.now()
		time *= (s - c).total_seconds()
	return time**(1/experiment_numbers)

def ProcessInput6(x1, x2, experiment_numbers): 
	time = 1.0
	for i in range(experiment_numbers): 
		dictionary = {}
		c = datetime.datetime.now()
		result = KaratsubaMultiplyWithSpatialLocality_16(x1, x2, dictionary)
		s = datetime.datetime.now()
		time *= (s - c).total_seconds()
	return time**(1/experiment_numbers)

def main(): 
	 
	x1 = []
	x2 = []
	samples = 100
	for i in range(1, samples): 
		x1.append(decimal.Decimal(random.getrandbits(i)))
		x2.append(decimal.Decimal(random.getrandbits(i)))
	#timesA = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(ProcessInput2)(x1[i-1], x2[i-1]) for i in range(1, samples)))
	#timesB = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(ProcessInput3)(x1[i-1], x2[i-1]) for i in range(1, samples)))
	timesC = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(ProcessInput)(x1[i-1], x2[i-1], 25) for i in range(1, samples)))
	timesD = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(ProcessInput4)(x1[i-1], x2[i-1], 25) for i in range(1, samples)))
	timesE = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(ProcessInput5)(x1[i-1], x2[i-1], 25) for i in range(1, samples)))
	timesF = np.array(Parallel(n_jobs = 1)(delayed(ProcessInput6)(x1[i-1], x2[i-1]) for i in range(1, samples)))
	plt.xlabel('bits')
	plt.ylabel('time/s')
	#a, = plt.plot(np.arange(1, samples, 1), timesA[0:,], label = 'normal multiplication')
	#b, = plt.plot(np.arange(1, samples, 1), timesB[0:,], label = 'Karatsuba multiplication')
	c, = plt.plot(np.arange(1, samples, 1), timesC[0:,], label = 'Karatsuba multiplication with memoisation')
	d, = plt.plot(np.arange(1, samples, 1), timesD[0:,], label = 'Karatsuba multiplication with 2-way set associative locality')
	e, = plt.plot(np.arange(1, samples, 1), timesE[0:,], label = 'Karatsuba multiplication with 2-way set associative locality')
	f, = plt.plot(np.arange(1, samples, 1), timesF[0:,], label = 'Karatsuba multiplication with 16-way set associative locality')
	plt.legend([c, d, e, f], ['karatsuba multiplication with memoisation', '3 stores on a miss', '6 stores on a miss', '32 stores on a miss'])
	#plt.plot(np.arange(0, 500, 1), timesC)
	plt.show()
	epsilon = 10**-8
	#plt.plot(np.array(timesA)/np.array(timesB) + epsilon)
	#plt.show()
	#print(np.mean(timesA)/np.mean(timesB))
if __name__ == "__main__": 
		main()