#Busra_Unlu_211711008_HW6

import numpy as np
import cv2 as cv
import imutils

img1=cv.imread("C:/Users/busra/BIL561/HW6/uni_test_3.jpg")
img2=cv.imread("C:/Users/busra/BIL561/HW6/uni_test_4.jpg")
img1 = imutils.resize(img1, width=700)
img2 = imutils.resize(img2, width=700)

def Panaroma(image1,image2,lowe_ratio=0.75, max_Threshold=4.0,val=0):
    lowe_ratio=0.75
    max_Threshold=4.0
    #FINDING KEYPOINTS AND FEATURES
    #go gray
    gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    #SIFT
    if val==0:
        #surf = cv.xfeatures2D.SURF_create ()   //not working 
        sift = cv.SIFT_create()
        #detecting 
        (kps1, f1) = sift.detectAndCompute(gray_image1, None)
        kps1 = np.float32([i.pt for i in kps1])
        (kps2, f2) = sift.detectAndCompute(gray_image2, None)
        kps2 = np.float32([i.pt for i in kps2])
        #MATCHING KEYPOINTS
        #with brute force and knnmatch
        matcher=cv.DescriptorMatcher_create("BruteForce") 
        #(cv.DescriptorMatcher_FLANNBASED) tried but got error so cannot be implemented
        rawMatches=matcher.knnMatch(f1,f2,2)

        matches = []
        for val in rawMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                matches.append((val[0].trainIdx, val[0].queryIdx))
       
        points1=np.float32([kps1[i] for (_,i) in matches])
        points2=np.float32([kps2[i] for (i,_) in matches])

        #estimating warping parameters with homography
        homography,status=cv.findHomography(points1,points2,cv.RANSAC,max_Threshold)

        value = image1.shape[1] + image2.shape[1]
        result_image = cv.warpPerspective(image1, homography, (value , image1.shape[0]))
        result_image[0:image2.shape[0], 0:image2.shape[1]] = image2
        return result_image

    #ORB
    if val==1:
        orb=cv.ORB_create()
        kps1, f1 =orb.detectAndCompute(gray_image1,None)
        kps1 = np.float32([i.pt for i in kps1])
        kps2, f2 =orb.detectAndCompute(gray_image2,None)
        kps2 = np.float32([i.pt for i in kps2])

        #MATCHING KEYPOINTS
        #with brute force and knnmatch
        matcher=cv.DescriptorMatcher_create("BruteForce")
        #(cv.DESCRIPTOR_MATCHER_BRUTEFORCE),cv.DescriptorMatcher_FLANNBASED

        rawMatches=matcher.knnMatch(f1,f2,2)

        matches = []
        for val in rawMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                matches.append((val[0].trainIdx, val[0].queryIdx))
       
        points1=np.float32([kps1[i] for (_,i) in matches])
        points2=np.float32([kps2[i] for (i,_) in matches])

        #estimating warping parameters with homography
        homography,status=cv.findHomography(points1,points2,cv.RANSAC,max_Threshold)

        value = image1.shape[1] + image2.shape[1]
        result_image = cv.warpPerspective(image1, homography, (value , image1.shape[0]))
        result_image[0:image2.shape[0], 0:image2.shape[1]] = image2
        return result_image
    
    #AKAZE 
    if val==2:
        akaze=cv.AKAZE_create()
        kps1, f1 =akaze.detectAndCompute(gray_image1,None)
        kps1 = np.float32([i.pt for i in kps1])
        kps2, f2 =akaze.detectAndCompute(gray_image2,None)
        kps2 = np.float32([i.pt for i in kps2])

        #MATCHING KEYPOINTS
        #with brute force and knnmatch
        matcher=cv.DescriptorMatcher_create("BruteForce")
        #(cv.DESCRIPTOR_MATCHER_BRUTEFORCE)cv.DescriptorMatcher_FLANNBASED
        rawMatches=matcher.knnMatch(f1,f2,2)

        matches = []
        for val in rawMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                matches.append((val[0].trainIdx, val[0].queryIdx))
       
        points1=np.float32([kps1[i] for (_,i) in matches])
        points2=np.float32([kps2[i] for (i,_) in matches])

        #estimating warping parameters with homography
        homography,status=cv.findHomography(points1,points2,cv.RANSAC,max_Threshold)

        value = image1.shape[1] + image2.shape[1]
        result_image = cv.warpPerspective(image1, homography, (value , image1.shape[0]))
        result_image[0:image2.shape[0], 0:image2.shape[1]] = image2
        return result_image

    #KAZE 
    if val==3:
        kaze=cv.KAZE_create()
        kps1, f1 =kaze.detectAndCompute(gray_image1,None)
        kps1 = np.float32([i.pt for i in kps1])
        kps2, f2 =kaze.detectAndCompute(gray_image2,None)
        kps2 = np.float32([i.pt for i in kps2])

        #MATCHING KEYPOINTS
        #with brute force and knnmatch
        matcher=cv.DescriptorMatcher_create("BruteForce")
        #(cv.DESCRIPTOR_MATCHER_BRUTEFORCE)cv.DescriptorMatcher_FLANNBASED
        rawMatches=matcher.knnMatch(f1,f2,2)

        matches = []
        for val in rawMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                matches.append((val[0].trainIdx, val[0].queryIdx))
       
        points1=np.float32([kps1[i] for (_,i) in matches])
        points2=np.float32([kps2[i] for (i,_) in matches])

        #estimating warping parameters with homography
        homography,status=cv.findHomography(points1,points2,cv.RANSAC,max_Threshold)

        value = image1.shape[1] + image2.shape[1]
        result_image = cv.warpPerspective(image1, homography, (value , image1.shape[0]))
        result_image[0:image2.shape[0], 0:image2.shape[1]] = image2
        return result_image
    

panaroma_img=Panaroma(img1,img2,val=0)
panaroma_img_1=Panaroma(img1,img2,val=1)
panaroma_img_2=Panaroma(img1,img2,val=2)
panaroma_img_3=Panaroma(img1,img2,val=3)

cv.imshow("panaroma image (SIRF)",panaroma_img)
cv.imshow("panaroma image (ORB)",panaroma_img_1)
cv.imshow("panaroma image (AKAZE)",panaroma_img_2)
cv.imshow("panaroma image (KAZE)",panaroma_img_3)

#Yapılan işlemlerde Flann ve surf oynatılamamıstır. 



cv.waitKey(0)