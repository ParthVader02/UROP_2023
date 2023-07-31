import ctypes
import socket
import numpy as np

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

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

#first braille
letters= ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z"]
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind((HOST, PORT))
		s.listen()
		conn, addr = s.accept()
		with conn:
			print(f"Connected by {addr}")
			while True:
				data = conn.recv(1024)
				if data == b'change':
					word = random_word(letters)
					clientLib.nvdaController_brailleMessage(word)
					conn.sendall(b"changeconfirm")
				if data== b"close":
					break
