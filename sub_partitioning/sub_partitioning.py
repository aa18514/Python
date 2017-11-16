import random 
import datetime
import numpy as np
import matplotlib.pyplot as plt

def timedelta_total_seconds(td):
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / float(10**6)

def permute(a, k):
	result = 0
	lists = []
	j = 0
	for i in range(len(a)):
			if(a[i] >= k):
				if(len(a[j:i]) > 1 and a[j] < k):
					lists.append(a[j:i])
					j = i
			else: 
				result +=  1
	if(len(lists) == 0): 
		lists.append(a)
	else: 
		lists.append(a[j:])
	for i in range(len(lists)):			
		if(min(lists[i]) >= k): 
			result += 0
		elif(sum(lists[i]) < k):
			length_a = len(lists[i])
			result += 0.5 * length_a * (length_a - 1)
		else: 
			for j in range(len(lists[i])):
				res = lists[i][j]
				p = j + 1
				while(p < len(lists[i])): 
					res += lists[i][p]
					if(res >= k): 
						break
					else:
						result += 1
						p += 1
	return result

if __name__ == "__main__":
	times = []
	n_input = []
	time = 10000;
	for j in range(1, 60):
			lists = []
			time = float(1)
			n_input.append(j)
			itera = []
			for iterations in range(210):
				for i in range(j):
						lists.append(random.randint(0, 6))
				a = datetime.datetime.now()
				result = permute(lists, 6)
				b = datetime.datetime.now()
				d = (b - a).total_seconds()
				itera.append(float(d))
				#full_time = datetime.timedelta(time * (b - a).total_seconds())
				#time = timedelta_total_seconds(full_time)
				#print(float(time))
			result = np.sum(np.array(itera))
			times.append(result/210)
	n_input = [x for x in n_input]
	plt.plot(n_input, times)
	plt.show()
	print(times)
	print(permute([1, 2, 3, 6, 7, 8, 1, 1], 6))
	print(permute([1, 1], 6))
	print(permute([1, 1, 2], 6))
	print(permute([1, 1, 2, 3], 6))
	print(permute([2, 1, 2, 4], 6))