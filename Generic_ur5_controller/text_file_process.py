import csv
with open("laws.txt","r") as temp_file:
    file = temp_file.read()
print(file)

spacetext = "" #initialise string of text with spaces
for i in file: # loops through every character in the text
    if i.isalpha() or i==" " or i=="\n": # removes any punctuation, numbers
        spacetext += i.lower() # adds the lowercase version of any letters
plaintext = spacetext.replace(" ","") # removes spaces
plaintext = plaintext.replace("\n","") # removes new lines
print(plaintext)

splitplain = [plaintext[i:i+20] for i in range(0, len(plaintext), 20)] # split into lines of 20 characters
for i in range(len(splitplain)): splitplain[i] += "\n" # add new line character to end of each line

plaintext_file = open("newlaws.txt","w") # write to new text file -> this file is put into braille reader SD card
plaintext_file.writelines(splitplain) # write each line
plaintext_file.close() # close file

wordno = len(spacetext.split()) # number of words
lettno = len(plaintext) # number of letters
lineno = len(splitplain) # number of lines, same as doing int(np.ceil(len(plaintext)/20))

props = [str(wordno),str(lettno),str(lineno)] # list of properties

with open('test_properties.csv', 'w') as f: #append string of letters, velocity and times to csv
    write = csv.writer(f)
    write.writerow(props) #write proerties to csv
print(wordno,lettno,lineno)