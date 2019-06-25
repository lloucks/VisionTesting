import cv2
import numpy as np


jiangshi = cv2.imread('images/Jiangshi_lg.png', 0)

scale_percent = 40

# width = int(jiangshi.shape[1] * scale_percent / 100)
# height = int(jiangshi.shape[0] * scale_percent /100)
# dim = (width,height)

# jiangshi = cv2.resize(jiangshi, dim, interpolation = cv2.INTER_AREA)

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(jiangshi,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

cap =cv2.VideoCapture(0)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    #frame = cv2.imread('images/test1.jpg')
    # Operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2,des2 = sift.detectAndCompute(frame,None)

    matches = flann.knnMatch(des1,des2,k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for i, (match1,match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            matchesMask[i] = [1,0]

    draw_params = dict(matchColor=(0,255,0),
                  singlePointColor=(255,0,0),
                  matchesMask=matchesMask,
                  flags=0)
    
    flann_matches = cv2.drawMatchesKnn(jiangshi,kp1,gray_frame,kp2,matches,None,**draw_params)

    width = int(flann_matches.shape[1] * scale_percent / 100)
    height = int(flann_matches.shape[0] * scale_percent /100)
    dim = (width,height)

    flann_matches = cv2.resize(flann_matches, dim, interpolation = cv2.INTER_AREA)
    # Show image
    cv2.imshow('frame', flann_matches)

    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break

cap.release()
cv2.destroyAllWindows()
