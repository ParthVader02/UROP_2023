import time
import ctypes
import socket
import numpy as np

def random_word(letters): # generate a random word of length 3
	word = ""
	for i in range(0, 3):
		word += letters[np.random.randint(0, len(letters) - 1)]
	return word

#Load the NVDA client library
clientLib=ctypes.windll.LoadLibrary('./nvdaControllerClient64.dll')

#Test if NVDA is running, and if its not show a message
res=clientLib.nvdaController_testIfRunning()
if res!=0:
	errorMessage=str(ctypes.WinError(res))
	ctypes.windll.user32.MessageBoxW(0,u"Error: %s"%errorMessage,u"Error communicating with NVDA",0)

#braille some messages
change_flag = False
letters= ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z"]
word = random_word(letters)
while True:
	if change_flag == False:
		clientLib.nvdaController_brailleMessage(word)
	