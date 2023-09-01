with open ('/home/parth/UROP_2023/predicted_text_0.1.txt', 'r') as f: # read in predicted text
    pred_text = f.read()
rows = pred_text.split('\n') # split into lines
rows = rows[:-1] # remove last element that is empty
with open ('/home/parth/UROP_2023/alphabetnew.txt', 'r') as f: # read in ground truth text
    gt_text = f.read()
gt_text = gt_text.split('\n') # split into lines
gt_text = gt_text[:-1] # remove last element that is empty

if len(gt_text[-1]) <20: # add spaces to end of last line if less than 20 characters
    end_space = 20 - len(gt_text[-1])
    gt_text[-1] += end_space*' ' 

block_size = round(len(rows[0])/len(gt_text[0])) # calculate block size
pos_count = 0
neg_count = 0

for i in range(len(gt_text)): # loop through each line in ground truth and predicted text
    gt_row = gt_text[i]
    pred_row = rows[i]
    pred_row = [pred_row[j:j+block_size] for j in range(0, len(pred_row), block_size)] # split into blocks of block_size characters that spaced by blocksize
    print(pred_row)
    if len(pred_row) > 20:
        #print(gt_row, pred_row)
        pred_row = pred_row[:-1] # remove last element if extra element is present

    for k in range(0,len(pred_row)):
        print(pred_row[k], gt_row[k])
        if gt_row[k] in pred_row[k]:
            pos_count += 1
        else:
            neg_count += 1
print(pos_count, neg_count)
print(pos_count/(pos_count+neg_count))
