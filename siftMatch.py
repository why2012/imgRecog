#coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

def filter_matches(kp1, kp2, matches, ratio = 0.75):  
	mkp1, mkp2 = [], []  
	for m in matches:  
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:  
			m = m[0]  
			mkp1.append(kp1[m.queryIdx])  
			mkp2.append(kp2[m.trainIdx])  
	p1 = np.float32([kp.pt for kp in mkp1])  
	p2 = np.float32([kp.pt for kp in mkp2])  
	kp_pairs = zip(mkp1, mkp2)  
	return p1, p2, kp_pairs  

def feature_matches(des1, des2, k = 2, ratio = 0.75):
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k = k)
	filtered_matches = []
	for m in matches:
		if len(m) >= 2 and m[0].distance < m[1].distance * ratio:
			filtered_matches.append(m)
	return filtered_matches

def feature_matches_inv(des1, des2, k = 2, ratio = 0.75):
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k = k)
	filtered_matches = []
	for m in matches:
		if len(m) >= 2 and m[0].distance >= m[1].distance * ratio:
			filtered_matches.append(m)
	return filtered_matches

# sm.plot_matched_linechart(matches)
def plot_matched_linechart(matches):
	for points in matches:
		plt.plot(range(1, len(points) + 1), [i.distance for i in points])
	plt.show()

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):  
	h1, w1 = img1.shape[:2]  
	h2, w2 = img2.shape[:2]  
	vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)  
	vis[:h1, :w1] = img1  
	vis[:h2, w1:w1 + w2] = img2  
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  
	  
	if H is not None:  
		corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])  
		corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))  
		cv2.polylines(vis, [corners], True, (255, 255, 255))  
  	
	if status is None:  
		status = np.ones(len(kp_pairs), np.bool)  
	p1 = np.int32([kpp[0].pt for kpp in kp_pairs])  
	p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
    
	green = (0, 255, 0)  
	red = (0, 0, 255)  
	white = (255, 255, 255)  
	kp_color = (51, 103, 236)  
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):  
		if inlier:  
			col = green  
			cv2.circle(vis, (x1, y1), 3, col, -1)  
			cv2.circle(vis, (x2, y2), 3, col, -1)  
		else:  
			col = red  
			r = 2  
			thickness = 3  
			cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)  
			cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)  
			cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)  
			cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)  
	vis0 = vis.copy()  
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):  
		if inlier:  
			cv2.line(vis, (x1, y1), (x2, y2), red)  
  
	cv2.imshow(win, vis)
	cv2.waitKey(10)

# imgFeature = cv2.imread("pics/IMG_1203.JPG")
# imgDest = cv2.imread("pics/IMG_1205_SRC.JPG")
# imgDest2 = cv2.imread("pics/IMG_1205_SRC2.JPG")
# imgDest3 = cv2.imread("pics/IMG_1205_SRC3.JPG") 
# matchResult=[];reload(siftM);siftM.siftTest(imgFeature, imgDest, matchResult)
# 放大目标图像，使目标图像与特征图像大小一致，匹配效果更好, 5倍最好
# 对于旋转不多的目标图像， 改用surf，加大octave，提取特征更快
def siftTest(imgFeature, imgDest, matchResult = [], drawBoundingBox = True):
	imgFeatureGray = cv2.cvtColor(imgFeature, cv2.COLOR_BGR2GRAY)
	imgDestGray = cv2.cvtColor(imgDest, cv2.COLOR_BGR2GRAY)
	if not matchResult:
		print "Detect..."
		# sift = cv2.xfeatures2d.SURF_create(nOctaveLayers = 8)
		# # [contrastThreshold]: 0.03, [edgeThreshold]: 10, [sigma]: 1.6
		sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 8, contrastThreshold = 0.01, edgeThreshold = 10, sigma = 1.6)
		(kpsFeatureImg, descsFeatureImg) = sift.detectAndCompute(imgFeature, None)
		(kpsDestImg, descsDestImg) = sift.detectAndCompute(imgDest, None)
		bf = cv2.BFMatcher(cv2.NORM_L2)
		matches = bf.knnMatch(descsFeatureImg, descsDestImg, k = 2)
		matchResult.extend([kpsFeatureImg, kpsDestImg, matches])
	else:
		kpsFeatureImg = matchResult[0]
		kpsDestImg = matchResult[1]
		matches = matchResult[2]
	print "Match..."
	p1, p2, kpPairs = filter_matches(kpsFeatureImg, kpsDestImg, matches, ratio = 0.9) # ratio = 0.5
	print "MatchResult: ", len(kpPairs)
	if kpPairs:
		if drawBoundingBox:
			# F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS)
			# # We select only inlier points
			# p1 = p1[mask.ravel() == 1]
			# p2 = p2[mask.ravel() == 1]
			M, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
			if M is not None:
				imgFeatureH, imgFeatureW = imgFeatureGray.shape
				pts = np.float32([[0, 0], [imgFeatureW - 1, 0], [imgFeatureW - 1, imgFeatureH - 1], [0, imgFeatureH - 1]])
				pts = pts.reshape(-1, 1, 2);
				dst = cv2.perspectiveTransform(pts, M)
				print M
				print pts
				print dst
				imgBoundingBox = cv2.polylines(imgDest.copy(), [np.int32(dst)], True, (0, 255, 0), 1, cv2.LINE_AA)
				cv2.imshow("bounding", imgBoundingBox)
		explore_match('matches', imgFeatureGray, imgDestGray, kpPairs) 

# 放大目标图像，5x，特征提取效果更好
def keypointsDetect(imgFeature, imgDest):
	# imgFeatureGray = cv2.cvtColor(imgFeature, cv2.COLOR_BGR2GRAY)
	# imgDestGray = cv2.cvtColor(imgDest, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 8, contrastThreshold = 0.01, edgeThreshold = 20)
	(kpsFeatureImg, descsFeatureImg) = sift.detectAndCompute(imgFeature, None)
	(kpsDestImg, descsDestImg) = sift.detectAndCompute(imgDest, None)
	print len(kpsFeatureImg), len(kpsDestImg)
	imgFeature = cv2.drawKeypoints(imgFeature.copy(), kpsFeatureImg, None, color = (0, 0, 255))
	imgDest = cv2.drawKeypoints(imgDest.copy(), kpsDestImg, None, color = (0, 0, 255))
	cv2.imshow("imgFeature", imgFeature)
	cv2.imshow("imgDest", imgDest)
	cv2.waitKey(10)
	return [(kpsFeatureImg, descsFeatureImg), (kpsDestImg, descsDestImg)]

def drawKeypoints(imgDest, points, matches = None):
	if matches is not None:
		pindex = [i.queryIdx for i in matches[:, 0]]
		points = points[pindex]
		imgDest = cv2.drawKeypoints(imgDest.copy(), points, None, color = (0, 0, 255))
	else:
		imgDest = imgDest.copy()
		for points1D in points:
			color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
			for p in points1D:
				imgDest = cv2.circle(imgDest, (int(p[0]), int(p[1])), 4, color, 2)
	cv2.imshow("points", imgDest)
	cv2.waitKey(10)

def drawKnnMatches(imgFeature, featurePoints, imgDest, destPoints, matches):
	img = cv2.drawMatchesKnn(imgFeature, featurePoints, imgDest, destPoints, matches, None)
	cv2.imshow("points", img)
	cv2.waitKey(10)
	# plt.imshow(img3,)
	# plt.show()

# features = sm.keypointsDetect(featureImg, destImg)
# result = sm.feature_matches(features[0][1], features[1][1], k = 20, ratio = 0.75)
# originP, destP = sm.drawBoundingBox(featureImg, np.array(features[0][0]), destImg, np.array(features[1][0]), np.array(result)[:, 0:6].reshape(-1, 6), eps = 140)
# sm.drawKeypoints(destImg, destP)
def drawBoundingBox(imgFeature, featurePoints, imgDest, destiPoints, matches, eps = 50):
	imgBoundingBox = imgDest.copy()
	imgFeatureGray = cv2.cvtColor(imgFeature, cv2.COLOR_BGR2GRAY)
	imgFeatureH, imgFeatureW = imgFeatureGray.shape
	_, roundNum = matches.shape
	originPoints = []
	destPoints = []
	rawOriginPoints = []
	rawDestPoints = []
	for roundIndex in range(roundNum):
		queryIdx = [i.queryIdx for i in matches[:, roundIndex]]
		trainIdx = [i.trainIdx for i in matches[:, roundIndex]]
		rawOriginPoints.extend([p.pt for p in featurePoints[queryIdx]])
		rawDestPoints.extend([p.pt for p in destiPoints[trainIdx]])
	rawOriginPoints = np.array(rawOriginPoints)
	rawDestPoints = np.array(rawDestPoints)
	db = DBSCAN(eps = eps, min_samples = 1).fit(rawDestPoints)
	# db = KMeans(n_clusters = 4).fit(rawDestPoints)
	labelSet = set(db.labels_)
	print len(labelSet)
	for l in labelSet:
		originPoints.append(rawOriginPoints[db.labels_ == l])
		destPoints.append(rawDestPoints[db.labels_ == l])
		# print rawDestPoints[db.labels_ == l]
	for roundIndex in range(len(labelSet)):
		M, mask = cv2.findHomography(np.float32(originPoints[roundIndex]), np.float32(destPoints[roundIndex]), cv2.RANSAC)
		print M
		if M is not None:
			pts = np.float32([[0, 0], [imgFeatureW - 1, 0], [imgFeatureW - 1, imgFeatureH - 1], [0, imgFeatureH - 1]])
			pts = pts.reshape(-1, 1, 2);
			dst = cv2.perspectiveTransform(pts, M)
			imgBoundingBox = cv2.polylines(imgBoundingBox, [np.int32(dst)], True, (0, 255, 0), 1, cv2.LINE_AA)
	cv2.imshow("bounding", imgBoundingBox)
	cv2.waitKey(10)
	return originPoints, destPoints

def match(featureImg, destImg, k = 20, eps = 110, useK = 10, matchRatio = 0.75):
	features = keypointsDetect(featureImg, destImg)
	result = feature_matches(features[0][1], features[1][1], k = k, ratio = matchRatio)
	originP, destP = drawBoundingBox(featureImg, np.array(features[0][0]), destImg, np.array(features[1][0]), np.array(result)[:, 0: useK].reshape(-1, useK), eps = eps)
	drawKeypoints(destImg, destP)






