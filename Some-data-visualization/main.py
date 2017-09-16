import numpy as np 
import matplotlib.pyplot as plt 
import plot
from bs4 import BeautifulSoup
import re 
from collections import Counter 
import itertools
import pandas as pd

stars = [np.zeros(5, ) for i in range(8)]
ratings = dict()
train_data = []
test_data = [] 

def read_book_ratings(book_name, book_index): 
	with open(book_name, 'r', encoding="utf-8") as csvfile: 
			lines = csvfile.readlines()
			i = 0 
			for line in lines:  
					rating = float(line[:3])
					soup = BeautifulSoup(line[3:], "html.parser")
					spans = soup.find_all('span', attrs={'class': ''})
					if(book_index == 3 and i < 20000):
							if(rating not in ratings):
								ratings[int(rating)] = 1
							else: 
								ratings[int(rating)] += 1
							train_data.append(spans[0].text)
							i = i + 1
					elif(book_index == 3 and i >= 20000):
							test_data.append(spans[0].text)
							i = i + 1 
					stars[book_index][int(rating - 1)] += 1 

if __name__ == "__main__":
		read_book_ratings('Donna-Tartt-The-Goldfinch.csv', 0)
		read_book_ratings('El-James-Fifty-Shades-of-Grey.csv', 1)
		read_book_ratings('Andy-Weir-The-Martian.csv', 2)
		read_book_ratings('Fillian_Flynn-Gone_Girl.csv', 3)
		read_book_ratings('John-Green-The-Fault-in-our-Stars.csv', 4)
		read_book_ratings('Laura-Hillenbrand-Unbroken.csv', 5)
		read_book_ratings('Paula_Hawkins-The-Girl-On-The-Train.csv', 6)
		read_book_ratings('Suzanne-Collins-The-Hunger-Games.csv', 7)
		plot.plot(stars)
		list_bow = []
		BOW_df = pd.DataFrame(0, columns=ratings, index = '')
		words_set = Set()
		for i in range(0, len(train_data)): 
			pat = re.compile(r"([.()!])")
			train_data[i] = pat.sub(" \\1 ", train_data[i])
			pat = re.compile(r"(['['])")
			train_data[i] = pat.sub(" \\1 ", train_data[i])
			pat = re.compile(r"([']'])")
			train_data[i] = pat.sub(" \\1 ", train_data[i])
			pat = re.compile(r"([,()!])")
			train_data[i] = pat.sub(" \\1 ", train_data[i])	
			pat = re.compile(r"( [(] )")
			train_data[i] = pat.sub(" \\1 ", train_data[i])
			pat = re.compile(r"( [)] )")
			train_data[i] = pat.sub(" \\1 ", train_data[i])
			pat = re.compile(r"([:()!])")
			train_data[i] = pat.sub(" \\1 ", train_data[i])
			pat = re.compile(r"([;()!])")
			train_data[i] = pat.sub(" \\1 ", train_data[i])
			result = train_data[i]
			score = ratings[i]
			splitted_text = result.split()
			for word in splitted_text: 
				if word not in words_set:
					words_set.add(word)
					BOW_DF.loc[word] = [0,0,0,0,0]
					BOW_DF.ix[word][score] += 1 
				else:
					BOW_DF.ix[word][score] += 1
		print(BOW_DF)