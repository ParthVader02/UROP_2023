import cv2
import numpy as np
from scipy.spatial import KDTree

# gives the image, combined and points only for 7 different images

thresh = 0.02
mysigma = 0.8

imgs,ptsimg,combinedimg = [], [], []
for i in range(10):
    imgs.append(cv2.imread("raw_data/im"+str(i+1)+".jpg"))
    ptsimg.append(np.ones(imgs[i].shape, np.uint8) * 0xFF)
    combinedimg.append(cv2.imread("raw_data/im"+str(i+1)+".jpg"))

sift = cv2.SIFT_create(contrastThreshold=thresh, edgeThreshold=10, sigma=mysigma)
    #contrastThreshold basically controls the number of keypoints detected by strength
    #edgeThreshold doesnt have much effect so kept at default
    #sigma controls gaussian blur, reduced from default because camera not very strong, keep this constant

for k in range(10):
    kp = sift.detect(imgs[k],None)
    pts =  [key_point.pt for key_point in kp]
    pts = np.array(pts)
    pts_by_row = [[],[],[]]

    # any points too high up or too low down can't be part of a braile dot, so
    # should be removed. there are also 3 extra points ive spotted that it
    # frequently adds which are also removed
    toremove = []
    print(pts)

    for i in pts:
        if i[1] >= 110 and i[1] <= 150:
            pts_by_row[0].append(i)
        elif i[1] >= 160 and i[1] <= 190:
            pts_by_row[1].append(i)
        elif i[1] >= 205 and i[1] <= 235:
            pts_by_row[2].append(i)

    for i in range(3):
        pts_by_row[i] = np.array(pts_by_row[i])
    for i in pts_by_row: print(i)

    trees = []
    for i in range(3):
        if len(pts_by_row[i]) > 0:
            trees.append(KDTree(pts_by_row[i]))
        else:
            trees.append([])

    for i in range(3):
        for j in range(len(pts_by_row[i])):
            ds, inds =  trees[i].query(pts_by_row[i][j],k=range(2,11),p=2, distance_upper_bound=50) #get 3 nearest neighbours
            print(ds, inds)
            #for j in range(1,len(inds)):
            #    if ds[j] < 1: #remove points that are too close
            #        new_pt  = (pts[i] + pts[inds[j]])/2
            #        #print(new_pt)
            #        pts[i] = new_pt
            #        np.delete(pts, inds[j])

    for i in range(len(pts)):

        # cv2.circle(ptsimg[k], (120,int(pts[i][1])), 1, (0,0,255), -1)
        cv2.circle(ptsimg[k], (int(pts[i][0]),int(pts[i][1])), 5, (0,0,255), -1)
        cv2.circle(combinedimg[k], (int(pts[i][0]),int(pts[i][1])), 5, (0,0,255), -1)

# cv2.imshow("img", ptsimg)
# cv2.imshow("img", img)
toshow = []
for i in range(10):
    toshow.append(np.concatenate([imgs[i], combinedimg[i], ptsimg[i]], axis=0))

finalshow = np.concatenate([i for i in toshow], axis=1)
cv2.imshow("img",finalshow)
cv2.waitKey(0)


# this bit was for finding the best thresh and mysigma
'''
threshes = [0.006,0.008,0.01,0.012,0.014,0.016,0.018]
sigmas = [0.6,0.8,1]

# get to choose the image all by yourself, aren't you lucky
img = cv2.imread("sharp/im"+input()+".jpg")
ptsimg = [[np.ones(img.shape, np.uint8) * 0xFF for _ in range(len(threshes))] for _ in range(len(sigmas))]

for j in range(len(ptsimg)):
    for k in range(len(ptsimg[j])):
        sift = cv2.SIFT_create(contrastThreshold=threshes[k], edgeThreshold=10, sigma=sigmas[j])
        #contrastThreshold basically controls the number of keypoints detected by strength
        #edgeThreshold doesnt have much effect so kept at default
        #sigma controls gaussian blur, reduced from default because camera not very strong, keep this constant

        kp = sift.detect(img,None)
        pts =  [key_point.pt for key_point in kp]
        pts = np.array(pts)

        toremove = []
        for i in range(len(pts)):
            if pts[i][1] < 110 or pts[i][1] > 245:
                toremove.append(i)
        pts = np.delete(pts,toremove,0)
        print(pts)

        for i in range(len(pts)):
            cv2.circle(ptsimg[j][k], (int(pts[i][0]),int(pts[i][1])), 1, (0,0,255), -1)

toshow = []
for j in range(len(ptsimg)):
    toshow.append(np.concatenate([k for k in ptsimg[j]], axis=1))
    print(toshow[j])
finalshow = np.concatenate([i for i in toshow],axis=0)
print(finalshow)
cv2.imshow("img",finalshow)
cv2.waitKey(0)
'''
