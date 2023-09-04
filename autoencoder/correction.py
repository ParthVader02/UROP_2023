from math import *
import csv

# things required to run this:
# the predicted text in predictedtoworkwith.txt
# the true text, split into lines of 20, in plaintext.txt (can be created by
#       running converttoplain.py and having the normal plaintext with
#       punctuation etc in originaltext.txt)
# the confusion matrix in confusion_matrix.csv

#CONFUSION MATRIX
with open("/home/parth/UROP_2023/confusion_matrix_new.csv") as csvmatrix:
    confusion = list(csv.reader(csvmatrix, delimiter=","))

# removing the header row/column
confusion.pop(0)
for i in confusion:
    i.pop(0)

def confusion_value(orig,pred):
    # e.g. if original=p, predicted=l, returns confusion[15,11]
    # probability of true p being read as false l
    first = 26 if pred==" " else ord(pred)-97
    second = 26 if orig==" " else ord(orig)-97
    return float(confusion[first][second])

def constrained_partitions(n, k, min_elem, max_elem):
    # returns all lists of length k that sum
    # to n, and the elements in the list are restricted by min_elem, max_elem
    allowed = range(max_elem, min_elem-1, -1)

    def helper(n, k, t):
        if k == 0:
            if n == 0:
                yield t
        elif k == 1:
            if n in allowed:
                yield t + (n,)
        elif min_elem * k <= n <= max_elem * k:
            for v in allowed:
                yield from helper(n - v, k - 1, t + (v,))

    return list(helper(n, k, ()))

#GET DATA
with open("predicted_text_0.2.txt","r") as pr:
    predicted = pr.readlines()
with open("lawsnew.txt","r") as pla:
    plain = pla.readlines()

for i in range(len(predicted)): predicted[i] = predicted[i].replace("\n","")
for i in range(len(plain)): plain[i] = plain[i].replace("\n","")


# used at the end
all_lines = []
# all_scores = []
#DATA REDUCTION + ERROR CORRECTION
for myindex in range(len(predicted)):
    pl = predicted[myindex] # predicted line
    n = len(pl)/20 # factor by which predicted line is too long

    low_bl = floor(n)
    upp_bl = ceil(n)

    goal_length = len(plain[myindex])

    # if we're working on the last line in the text, the final bits will be empty
    # space so won't need to be read. this keeps the correct proportion of letters
    if myindex == len(predicted)-1:
        pl = pl[:ceil(n*goal_length)]


    # all_info has:
    # key is index of first letter and length of block
    # value is the block, letter it's likely to be, probability of it being correct
    all_info = {}

    # both the cases for min being used in for loops is so that it doesn't mess up
    # towards the end of the line with attempting read too much

    for i in range(len(pl)+1-low_bl): # i is index of start letter of a block

        for k in range(low_bl,min(upp_bl+1, len(pl)-i+1)): # k is block size
            # for one block, this dictionary contains all the possible true
            # letters in a block and the score assuming that letter is true
            score = {}

            for letter in set(pl[i:i+k]): # letter is the one assumed to be true in the block
                score[letter] = 0

                for j in range(min(k,len(pl)-i)): # j is index of letter in the block
                    score[letter] += confusion_value(letter,pl[i+j])

                score[letter] /= k

            highest_key = max(score,key=score.get)
            all_info[ (i,k) ] = [pl[i:i+k],highest_key,score[highest_key]]

    # goes through different possible combinations of blocks and calculates the
    # overall score of each partition

    all_parts = constrained_partitions(len(pl),goal_length,low_bl,upp_bl)
    all_part_scores = {} # will have the partition as key, average score as value
    for part in all_parts:
        myblocks = [(sum(part[:i]),part[i]) for i in range(goal_length)]
        my_part_score = 0
        for i in range(goal_length):
            my_part_score += all_info[myblocks[i]][2] # the score of that block
        my_part_score /= goal_length
        all_part_scores[part] = my_part_score

    best_part = max(all_part_scores,key=all_part_scores.get)
    final_blocks = [(sum(best_part[:i]),best_part[i]) for i in range(goal_length)] # list of tuples of (index,length) of blocks


    final_string= ""
    for i in final_blocks:
        final_string += all_info[i][1] # the most probably true letter


    # final_score = 0
    # for i in range(len(final_string)):
    #     if final_string[i]==plain[myindex][i]: final_score += 1
    # final_score /= len(final_string)
    all_lines.append(final_string)
    # all_scores.append(final_score)

    # print("length of predicted:",len(pl),"\nlength of true:",goal_length,"\nn:",n,"\nbounds for block length:",low_bl,upp_bl)
    # print(pl)
    # print(final_string)
    # print(plain[myindex])
    # print(final_score)
    # print()

# print(all_lines)
# print(all_scores)

complete_text = "".join(all_lines)

# END OF DATA REDUCTION STEP
########################################################################
# START OF SCORING

# calculating exact-match score
overall_score = 0
for i in range(len(complete_text)):
    if "".join(plain)[i] == complete_text[i]:
        overall_score += 1
overall_score /= len(complete_text)

# calculating lenient score
len_score = 0
for i in range(len(complete_text)):
    if i == 0 and "".join(plain)[i] in complete_text[:2]:
        len_score += 1
    elif i == 1 and "".join(plain)[i] in complete_text[:3]:
        len_score += 1
    elif "".join(plain)[i] in  complete_text[i-2:i+1]:
        len_score += 1
len_score /= len(complete_text)

print("Length of true text:", len("".join(plain)))
print("Length of predicted text:", len("".join(predicted)))
print("Length of a predicted line:", len(predicted[0]))
print("n:", n)
print("".join(predicted))
print("".join(plain))
print(complete_text)
print("Exact match:  ",overall_score)


print("Lenient match:",len_score)


# abcdefg
# aabcdeg
