import paho.mqtt.client as mqtt
import time
import struct
import json  
import logging
import argparse
from msvcrt import getch
from threading import Thread 
import sys
import msvcrt

class magnetic_flux(Thread): 
	def __init__(self, mqttc):
		super(magnetic_flux, self).__init__()
		self.__mqttc = mqttc

	def run(self): 
			self.__mqttc.on_message = self.recieve_payload
			self.__mqttc.loop_forever()
	
	def recieve_payload(self, topic, userdata, msg):  
		json_formatted_msg = msg.payload.decode("utf-8")
		print(json_formatted_msg)
		if( "Sensor faulty" in json_formatted_msg): 
			logging.warning(json_formatted_msg)
		else: 
			if("Sensor" in json_formatted_msg): 
				logging.info(json_formatted_msg)
			elif("on" in json_formatted_msg): 
				logging.info("device turned on" + json_formatted_msg[3:])
			elif("off" in json_formatted_msg):
				logging.info("device turned off" + json_formatted_msg[4:])

if __name__ == "__main__":
	"""detects if the device is functioning correctly
	and stores the when each device was turned on and off
	"along with the timestamp"""
	logging.basicConfig(filename = 'readings and status.log', filemode = 'w', level=logging.INFO)
	channels = [mqtt.Client(), mqtt.Client()] #there are two channels, one channel is to recieve the status of the sensor, and one channel is to transfer the times at which the specific device was turned off and turned on
	channels[0].connect('192.168.0.10', 1883, 60) 
	channels[0].subscribe('esys/SenSa/status')
	channels[1].connect('192.168.0.10', 1883, 60)
	channels[1].subscribe('esys/SenSa/reading')
	mf_1 = magnetic_flux(channels[0])
	mf_2 = magnetic_flux(channels[1])
	mf_1.start() 
	mf_2.start()
	mf_1.join()
	mf_2.join()
	