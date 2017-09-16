from machine import Pin, I2C 
import utime 
import json 
from clock import *
import ustruct as struct
from mqtt import * 

class i2c():	#12c class with functions to interface with the sensor
	def __init__(self, Pin1, Pin2, frequency):
		self.__ic =  I2C(scl = Pin(Pin1), sda = Pin(Pin2), freq = frequency)
		self.__slave_address = self.__ic.scan()[0]

	def set_sensor_mode(self, measurement_mode,gain,operating_mode):
		self.__ic.writeto_mem(self.__slave_address, 0, measurement_mode)
		self.__ic.writeto_mem(self.__slave_address, 1, gain)
		self.__ic.writeto_mem(self.__slave_address, 2, operating_mode)
		utime.sleep_ms(500)

	def read_data (self):
		self.__data = self.__ic.readfrom_mem(self.__slave_address,3,6)
		self.__x = struct.unpack('>h', self.__data[0:2])[0]
		self.__y = struct.unpack('>h', self.__data[2:4])[0]
		self.__z = struct.unpack('>h', self.__data[4:6])[0]
		self.__magMilliGauss = (((self.__x)**2 + (self.__z)**2 + (self.__y)**2)**0.5)*0.92  #for gain = 1 

	def test_mode(self):
		testStatus = "Sensor working "
		IC.set_sensor_mode(b"\x71",b"\xA0",b"\x00")
		for n in range(0,20): 
			IC.read_data()
			utime.sleep_ms(67)
			if (243 < IC.__x < 575 and 243 < IC.__z < 575 and 243 < IC.__y < 575): #range from data sheet for gain 5
				pass
			else:
				testStatus = "Sensor faulty "	
		return testStatus	

IC = i2c(5, 4, 50000)
net = Network('192.168.0.10', 'asdid', 60)		#connect to network
net.init_wlan_and_client()	
net.recieve_message(b"esys/time")				#wait for time
CLK = Clock(json.loads(net.__msg)['date'])		
net.publish_status("Sensor turned on")

testStatus = IC.test_mode()		# test if sensor working properly
time = CLK.get_time()
net.publish_status(testStatus + time)		

IC.set_sensor_mode(b"\x64",b"\x20",b"\x00") #normal mode 1.5 Hz, gain = 1, continuos measurement

turnedOff = True 	#assume device originally turned off

magnitudeSum = 0 	#calibrate with initial magnetic field
for n in range(0,3):
	IC.read_data() 
	magnitudeSum += IC.__magMilliGauss
	utime.sleep_ms(1500)

oldMag = magnitudeSum/3

#if the difference between the values is above some threshold the appliance is turned on
while True: 	
	magnitudeSum = 0 
	for n in range(0,3):
		IC.read_data() 
		magnitudeSum += IC.__magMilliGauss
		utime.sleep_ms(1500)

	magAvg = magnitudeSum/3
		
	time = CLK.get_time()
	if (turnedOff == True and magAvg - oldMag > 60):
		turnedOff = False
		net.publish_reading("on " + time)

	elif (turnedOff == False and magAvg - oldMag < -60):
		turnedOff = True
		net.publish_reading("off " + time)
	oldMag = magAvg
	
	