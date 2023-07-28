import time
import ctypes

#Load the NVDA client library
clientLib=ctypes.windll.LoadLibrary('./nvdaControllerClient64.dll')

#Test if NVDA is running, and if its not show a message
res=clientLib.nvdaController_testIfRunning()
if res!=0:
	errorMessage=str(ctypes.WinError(res))
	ctypes.windll.user32.MessageBoxW(0,u"Error: %s"%errorMessage,u"Error communicating with NVDA",0)

#Speak and braille some messages
words= ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
for word in words:
	clientLib.nvdaController_brailleMessage(word)
	time.sleep(1)
#clientLib.nvdaController_brailleMessage(u"Test completed!")
