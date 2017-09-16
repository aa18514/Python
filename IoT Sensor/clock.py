from machine import RTC
class Clock(): 
	def __init__(self, message):
		#parse time recieved from the broker 
		date_time = message.split(" ")
		date = date_time[0].split('-')
		time = date_time[1].split(":")
		#use rtc to keep track of time on board
		self.__rtc = RTC()
		self.__rtc.datetime((int(date[0]), int(date[1]), int(date[2]), 0, int(time[0]), int(time[1]), int(time[2][0:2]), 0)) 
		
	def get_time(self):
		#return time as string
		time = self.__rtc.datetime() 
		return ("%s-%s-%s %s:%s:%s") % (str(time[0]), str(time[1]), str(time[2]), str(time[4]), str(time[5]), str(time[6]))