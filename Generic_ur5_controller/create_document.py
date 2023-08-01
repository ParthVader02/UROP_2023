import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

string_length = 1000
rand_string = get_random_string(string_length)

text_file = open("training.txt", "w")
text_file.write(rand_string)
text_file.close()