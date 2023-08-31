from string import ascii_lowercase as alc
import pandas as pd
import numpy as np

def generate_text_file():
    alphabet = alc + " "
    text = ""
    for i in alphabet:
        #print(i)
        for j in range(60):
            text +=i
    with open("confusion.txt","w") as f:
        f.write(text)

generate_text_file()