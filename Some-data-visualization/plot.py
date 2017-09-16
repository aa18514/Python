import numpy as np 
import matplotlib.pyplot as plt 

def plot(stars):
	total_dataset_size = np.sum(stars[0]) + np.sum(stars[1]) + np.sum(stars[2]) + np.sum(stars[3]) + np.sum(stars[4]) + np.sum(stars[5]) + np.sum(stars[6]) + np.sum(stars[7]) 
	print(total_dataset_size)

	for i in range(0, 8): 
		tot = np.sum(stars[i])
		stars[i] = [val/tot for val in stars[i]]
	ratings_1 = (stars[0][0], stars[1][0], stars[2][0], stars[3][0], stars[4][0], stars[5][0], stars[6][0], stars[7][0])
	ratings_2 = (stars[0][1], stars[1][1], stars[2][1], stars[3][1], stars[4][1], stars[5][1], stars[6][1], stars[7][1])
	ratings_3 = (stars[0][2], stars[1][2], stars[2][2], stars[3][2], stars[4][2], stars[5][2], stars[6][2], stars[7][2])
	ratings_4 = (stars[0][3], stars[1][3], stars[2][3], stars[3][3], stars[4][3], stars[5][3], stars[6][3], stars[7][3])
	ratings_5 = (stars[0][4], stars[1][4], stars[2][4], stars[3][4], stars[4][4], stars[5][4], stars[6][4], stars[7][4])
	ind = np.arange(8)
	width = 0.05
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, ratings_1, width, color='r', label = "1-star")
	rects2 = ax.bar(ind + width, ratings_2, width, color='y', label = "2-star")
	rects3 = ax.bar(ind + 2*width, ratings_3, width, color='b', label = "3-star")	
	rects4 = ax.bar(ind + 3*width, ratings_4, width, color='g', label = "4-star")
	rects5 = ax.bar(ind + 4*width, ratings_5, width, color='black', label = "5-star")	
	ax.set_xticks(ind + width/2)
	plt.legend()
	plt.xticks(rotation=10)
	fig.tight_layout()
	plt.ylabel('% of total viewers')
	ax.set_xticklabels(('The Goldfinch', 'Fifty Shades of Grey', 'The Martian', 'Gone Girl', 'Fault In Our Stars', 'Unbroken', 'Girl On The Train', 'Hunger Games'))
	print(total_dataset_size) 	
	plt.show()