with open("originaltext.txt","r") as temp_file:
    file = temp_file.read()

print(file)

# the next 6 lines are different in order to have a more accurate word count.
# entirely unnecessary if you're not fussed by word count
spacetext = ""
for i in file: # loops through every character in the text
    if i.isalpha() or i==" " or i=="\n": # removes any punctuation, numbers
        spacetext += i.lower() # adds the lowercase version of any letters
plaintext = spacetext.replace(" ","")
plaintext = plaintext.replace("\n","")

print(plaintext)

splitplain = [plaintext[i:i+20] for i in range(0, len(plaintext), 20)]
for i in range(len(splitplain)): splitplain[i] += "\n"

plaintext_file = open("plaintext.txt","w")
plaintext_file.writelines(splitplain)
plaintext_file.close()

wordno = len(spacetext.split()) # number of words
lettno = len(plaintext) # number of letters
lineno = len(splitplain) # number of lines [== int(np.ceil(len(plaintext)/20))]

print(wordno,lettno,lineno)
