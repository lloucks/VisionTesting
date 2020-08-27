import cv2
import numpy as np


def mp4_to_avi(src_dir, dst_dir):
    #src_dir = "1.mp4"
    #dst_dir = "2.avi"

    video_cap = cv2.VideoCapture(src_dir)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   
            int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
    video_writer = cv2.VideoWriter(dst_dir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size) 

    success, frame = video_cap.read()
    while success:
        video_writer.write(frame)
        success, frame = video_cap.read()

    return dst_dir


#jiangshi = cv2.imread('images/Jiangshi_lg.png', 0)
#vetalas = cv2.imread('images/Vetalas_lg.png', 0)
draugr = cv2.imread('images/Draugr_lg.png', 0)

scale_percent = 20

width = int(draugr.shape[1] * scale_percent / 100)
height = int(draugr.shape[0] * scale_percent /100)
dim = (width,height)

#jiangshi = cv2.resize(jiangshi, dim, interpolation = cv2.INTER_AREA)
#vetalas = cv2.resize(vetalas, dim, interpolation = cv2.INTER_AREA)
draugr = cv2.resize(draugr, dim, interpolation = cv2.INTER_AREA)

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(draugr,None)
#kp2,des2 = sift.detectAndCompute(vetalas,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

#cap =cv2.VideoCapture(0)
src_dir='2019-08-02/00091.MTS'
dst_dir='2019-08-02/00091.avi'
#cap = cv2.VideoCapture(mp4_to_avi(src_dir, dst_dir))
cap = cv2.VideoCapture(src_dir)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print "entering while loop"
while cap.isOpened():
    print "while cap.isOpened():"
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print "gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)"
    #cv2.imshow('frame', gray)
    #print "cv2.imshow('frame', gray)"
    #cv2.waitKey(0) 
    #print "cv2.waitKey(0) "

    
    # Operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2,des2 = sift.detectAndCompute(gray_frame,None)

    #matches = flann.knnMatch(des1,des2,k=2)
    print "des2"
    print des2
    print "des2.shape"
    print np.array(des2).shape
    print "des1.shape"
    print np.array(des1).shape
    if des2 is None:
        continue

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for i, (match1,match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            matchesMask[i] = [1,0]
        

    draw_params = dict(matchColor=(0,255,0),
                  singlePointColor=(255,0,0),
                  matchesMask=matchesMask,
                  flags=0)

    
    flann_matches = cv2.drawMatchesKnn(draugr,kp1,gray_frame,kp2,matches,None,**draw_params)
    
    # Show image
    cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('detection', 600,600)
    cv2.imshow('frame', flann_matches)
    ratio = 0.7
    good = []
    for p, q in matches:
        if p.distance > q.distance*ratio:
            good.append(p)
    print "len(good)"
    print len(good)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break


cap.release()
cv2.destroyAllWindows()