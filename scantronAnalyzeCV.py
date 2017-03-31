#!/usr/bin/env python
#coding:utf-8
import cv2
import numpy as np
from sklearn.cluster import KMeans

def readImg(path):
	return cv2.imread(path)

def grayImg(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 图片宽度对应数组列数，高度对应数组行数
def getImgSize(img):
	return (img.shape[1], img.shape[0])

def swap(a, b):
	return b, a

# 根据宽高创建纯白图，宽高对应数组的列数和行数
def createWhiteImg(size):
	return np.uint8(np.ones((size[1], size[0])) * 255)

def createBlackImg(size):
	return np.uint8(np.zeros((size[1], size[0])))

def findContours(grayImg, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE):
	_, contours, hierarchy = cv2.findContours(grayImg, mode, method)
	return contours, hierarchy

def drawContours(img, contours, color = (0, 0, 0), thickness = -1):
	cv2.drawContours(img, contours, -1, color, thickness)  
	return img

def binaryInv(grayImg):
	thresh, bimg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU)
	return bimg

def binaryThresh(grayImg, thresh1 = 0, thresh2 = 255, mode = cv2.THRESH_BINARY):
	thresh, bimg = cv2.threshold(grayImg, thresh1, thresh2, mode)
	return bimg

def getKernel(size):
	kernel = np.uint8(np.zeros(size))
	kw, kh = kernel.shape
	for x in range(kh):  
		kernel[x, kw / 2] = 1 
	for x in range(kw):  
		kernel[kh / 2, x] = 1 
	return kernel

# 将矩形定点按top-left, top-right, bottom-right, bottom-left顺序排列
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def erosion(grayImg, kernel = getKernel((4, 4)), iterations = 1):
	return cv2.erode(grayImg, kernel, iterations = iterations)

def dilation(grayImg, kernel = getKernel((4, 4)), iterations = 1):
	return cv2.dilate(grayImg, kernel, iterations = iterations)  

def sobel(grayImg):
	gradX = cv2.Sobel(grayImg, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradY = cv2.Sobel(grayImg, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)
	return gradient

def showImg(*imgs):
	index = 0
	for img in imgs:
		cv2.imshow("img" + str(index), img)
		index += 1
	cv2.waitKey(10)
	cv2.destroyAllWindows()

def showImgSingle(img, wname = "img"):
	cv2.imshow(wname, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# 获取轮廓的包围矩形
def getBoundingRect(contours):
	boundingRect = []
	for contour in contours:
		boundingRect.append(cv2.boundingRect(contour))
	return boundingRect

# lines[ line[ subline[]], ...]
def drawLines(img, lines, color = (0, 0, 0), thickness = 1):
	for line in lines:
		for subline in line:
			cv2.line(img, (subline[0], subline[1]), (subline[2], subline[3]), color, thickness)

# boundingBox = [x, y, w, h] -> [x1, y1, x2, y2]
def convertBoundingBoxToBox(boundingBox):
	realBox = []
	for box in boundingBox:
		realBox.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
	return realBox

def drawBox(img, boxes, color = (0, 0, 0), thickness = 1):
	for box in boxes:
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)

def drawRectP4(img, points, color = (0, 0, 0), thickness = 1):
	lines = [[points[0], points[1]], [points[1], points[2]], [points[2], points[3]], [points[3], points[0]]]
	for line in lines:
		cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color, thickness)

# 从极坐标获取线段端点
def getLinesFromPolarCoord(polarLines, thresh = 4000):
	lines = []
	for rhoThetaPack in polarLines:
		subline = []
		for rho,theta in rhoThetaPack:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a * rho
		    y0 = b * rho
		    x1 = int(x0 + thresh * (-b))
		    y1 = int(y0 + thresh * (a))
		    x2 = int(x0 - thresh * (-b))
		    y2 = int(y0 - thresh * (a))
		    subline.append([x1, y1, x2, y2])
		lines.append(subline)
	return lines

# 找出面积大于一定阈值的矩形
def findMainBox(boundingBox, size, thresh = 2):
	boxes = []
	S = size[0] * size[1]
	for box in boundingBox:
		for x1, y1, x2, y2 in box:
			w = np.abs(x2 - x1)
			h = np.abs(y2 - y1)
			s = w * h
			if s > S / thresh:
				boxes.append(box)
	return boxes			

# 将短线段加长, d == 0(垂直)
def expandLine(lines, thresh):
	newLines = []
	if lines is None:
		return newLines
	for sublines in lines:
		newSublines = []
		for x1, y1, x2, y2 in sublines:
			if x1 > x2:
				x1, x2 = swap(x1, x2)
				y1, y2 = swap(y1, y2)
			newLine = [0] * 4 
			d1 = float(x2 - x1)
			d2 = float(y2 - y1)
			# 垂直
			if d1 >= -5 and d1 <= 5:
				newLine = [x1, y1, x2, y2]
				f = newLine[1] - newLine[3]
				f = int(float(f) / np.abs(f))
				newLine[1] = newLine[1] + f * thresh
				newLine[3] = newLine[3] - f * thresh
			else:
				deltaY1 = d2 / d1 * x1
				deltaY2 = d2 / d1 * thresh
				newLine[0] = 0
				newLine[1] = int(y1 - deltaY1)
				newLine[2] = int(x2 + thresh)
				newLine[3] = int(y2 + deltaY2)
			newSublines.append(newLine)
		newLines.append(newSublines)
	return newLines

# 保留一定角度的直线, 自动取角度绝对值
def remainLine(lines, angles = [[0, 10], [80, 90]]):
	if lines is None:
		return lines
	PI_DEGREE = 180.0 / np.pi
	newLines = []
	for sublines in lines:
		newSublines = []
		for x1, y1, x2, y2 in sublines:
			newLine = [x1, y1, x2, y2]
			if x1 - x2 == 0:
				theta = 90
			else:
				theta = np.arctan(np.abs(float(y1 - y2) / (x1 - x2))) * PI_DEGREE
			for angle in angles:
				if theta >= angle[0] and theta <= angle[1]:
					newSublines.append(newLine)
					break
		if len(newSublines) != 0:
			newLines.append(newSublines)
	return newLines

# 计算两直线夹角
def computeLineAngle(line1, line2):
	dx1 = line1[2] - line1[0]
	dx2 = line2[2] - line2[0]
	dy1 = line1[3] - line1[1]
	dy2 = line2[3] - line2[1]
	dx1 /= 100
	dx2 /= 100
	dy1 /= 100
	dy2 /= 100
	theta =  np.arccos((dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)))
	theta = theta * 180.0 / np.pi
	return theta

# 两条直线交点
# segment==True, 求线段交点, 允许thresh大小的误差
def computeIntersect(a, b, segment = False, thresh = 100):
	x1 = a[0]; y1 = a[1]; x2 = a[2]; y2 = a[3]; x3 = b[0]; y3 = b[1]; x4 = b[2]; y4 = b[3]  
	d = float((x1 - x2) * (y3 - y4)) - (y1 - y2) * (x3 - x4)
	if d != 0: 
		pt = [0, 0]
		pt[0] = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
		pt[1] = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d 
		if segment:
			x = [x1, x2, x3, x4]
			y = [y1, y2, y3, y4]
			x.remove(min(x))
			x.remove(max(x))
			y.remove(min(y))
			y.remove(max(y))
			x.sort()
			y.sort()
			x[0] -= thresh
			x[1] += thresh
			y[0] -= thresh
			y[1] += thresh
			if (pt[0] < x[0] or pt[0] > x[1]) or (pt[1] < y[0] or pt[1] > y[1]):
				pt = [-1, -1]
		return pt
	else:
		return [-1, -1]

# 线段交点
# thresh1可以调整交点超出边界的程度
def getIntersectPoints(lines, filterSize = None, thresh1 = 0):
	length = len(lines)
	points = []
	for i in range(length):
		for n in range(i, length):
			# 防止近似平行线相交
			angle = computeLineAngle(lines[i][0], lines[n][0])
			thresh0 = 60
			if angle <= thresh0 or angle >= 180 - thresh0:
				continue
			point = computeIntersect(lines[i][0], lines[n][0], True)
			if point[0] != -1:
				if filterSize:
					if point[0] >= 0 - thresh1 and point[0] <= filterSize[0] + thresh1 and point[1] >= 0 - thresh1 and point[1] <= filterSize[1] + thresh1:
						points.append(point)
				else:
					points.append(point)
	return points

# 边角点
def getBoundingCornerPoints(points, size):
	points = np.array(points)
	centerP = np.array([size[0] / 2.0, size[1] / 2.0])
	topLeftPoint = [[0, 0], 0]
	topRightPoint = [[size[0], 0], 0]
	bottomRightPoint = [[size[0], size[1]], 0]
	bottomLeftPoint = [[0, size[1]], 0]
	for point in points:
		length = np.sum((point - centerP) * (point - centerP))
		# topLeft
		if point[0] <= centerP[0] and point[1] <= centerP[1]:
			if length > topLeftPoint[1]:
				topLeftPoint[0] = point
				topLeftPoint[1] = length
		# topRight
		elif point[0] >= centerP[0] and point[1] <= centerP[1]:
			if length > topRightPoint[1]:
				topRightPoint[0] = point
				topRightPoint[1] = length
		# bottomRight
		elif point[0] >= centerP[0] and point[1] >= centerP[1]:
			if length > bottomRightPoint[1]:
				bottomRightPoint[0] = point
				bottomRightPoint[1] = length
		# bottomLeft
		else:
			if length > bottomLeftPoint[1]:
				bottomLeftPoint[0] = point
				bottomLeftPoint[1] = length

	return np.array([topLeftPoint[0], topRightPoint[0], bottomRightPoint[0], bottomLeftPoint[0]], dtype = np.float32)

# 求众数, 允许+-thresh误差, 若merge != 0, 则分别把+-前merge个连续符号的统计结果合并, 并且只返回合并后的+-两个结果
# [(elem, [count, [indexes]]), ...]
# merge: [([elem1, elem2, ...], [count, [indexes]]), ...]
def findMode(data, thresh = 1, merge = 2):
	# {[count, [indexes]]}
	elemMap = {}
	for index, elem in enumerate(data):
		finded = False
		if elem in elemMap:
			elemMap[elem][0] += 1
			elemMap[elem][1].append(index)
			finded = True
		elif thresh != 0:
			if elem + thresh in elemMap:
				elemMap[elem + thresh][0] += 1
				elemMap[elem + thresh][1].append(index)
				finded = True
			if elem - thresh in elemMap:
				elemMap[elem - thresh][0] += 1
				elemMap[elem - thresh][1].append(index)
				finded = True
		if not finded:
			elemMap[elem] = [1, [index]]
	# [(elem, [count, [indexes]])]
	elemMap = sorted(elemMap.items(), key=lambda e:e[1][0], reverse=True)
	if merge != 0:
		if merge < 0:
			merge = max(0, len(elemMap) + merge)
		newElemMap = [([], [0, []]), ([], [0, []])]
		for index, modeR in enumerate(elemMap):
			if modeR[0] >= 0:
				newElemMap[0][0].append(modeR[0])
				newElemMap[0][1][0] += modeR[1][0]
				newElemMap[0][1][1].extend(modeR[1][1])
			else:
				newElemMap[1][0].append(modeR[0])
				newElemMap[1][1][0] += modeR[1][0]
				newElemMap[1][1][1].extend(modeR[1][1])
			if index >= merge:
				break
		elemMap = newElemMap
	return elemMap

# 直线倾角
def findAngle(lines):
	angles = []
	if lines is None:
		return angles
	for sublines in lines:
		for x1, y1, x2, y2 in sublines:
			length = np.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2)) 
			angle = np.arcsin((y1 - y2) / length) * 180 / np.pi
			angles.append(np.round(angle))
	return angles

def getSlideWindow(arr, windowIndex, unitSize):
	if len(arr) < unitSize or windowIndex >= len(arr) or windowIndex < 0:
		return []
	if windowIndex + unitSize - 1 < len(arr):
		return arr[windowIndex: unitSize + windowIndex]
	else:
		windowArr = []
		windowArr.extend(arr[windowIndex: len(arr)])
		leftCount = unitSize - (len(arr) - windowIndex);
		windowArr.extend(arr[0: leftCount])
		return windowArr
# 畸变矫正幅度
def getSkewScale(topLeft, topRight, bottomRight, bottomLeft):
	topAvg = 0.5 * (topLeft[1] + topRight[1])
	bottomAvg = 0.5 * (bottomLeft[1] + bottomRight[1])
	leftAvg = 0.5 * (topLeft[0] + bottomLeft[0])
	rightAvg = 0.5 * (topRight[0] + bottomRight[0])
	# 四个点畸变值, 左上，右上，右下，左下
	# return [[topLeft[0] - leftAvg, topLeft[1] - topAvg], [topRight[0] - rightAvg, topRight[1] - topAvg], [bottomRight[0] - rightAvg, bottomRight[1] - bottomAvg], [bottomLeft[0] - leftAvg, bottomLeft[1] - bottomAvg]]
	# 只保留上方畸变值
	# return [[0, topLeft[1] - topAvg], [0, topRight[1] - topAvg], [0, topRight[1] - topAvg], [0, topLeft[1] - topAvg]]
	# 只保留平均畸变值
	A = ((topLeft[0] - leftAvg) + (topRight[0] - rightAvg)) / 2.0
	B = ((bottomRight[0] - rightAvg) + (bottomLeft[0] - leftAvg)) / 2.0
	C = ((topLeft[1] - topAvg) + (bottomLeft[1] - bottomAvg)) / 2.0
	D = ((topRight[1] - topAvg) + (bottomRight[1] - bottomAvg)) / 2.0
	# 取消X方向畸变, 针对图像内凹或外凸的情况, 平整图像不需要取消X方向畸变
	A = B = 0
	return [[A, C], [A, D], [B, D], [B, C]]

def determineBoxRatio(c1, c2, c3, c4, w, h, thresh = 0.1):
	w = float(w)
	h = float(h)
	c1 = np.array(c1)
	c2 = np.array(c2)
	c3 = np.array(c3)
	c4 = np.array(c4)
	c = [c1, c2, c3, c4]
	len1 = np.sum(c1 * c1)
	len2 = np.sum(c2 * c2)
	len3 = np.sum(c3 * c3)
	len4 = np.sum(c4 * c4)
	lenTotal = [len1, len2, len3, len4]
	minLenIndex = np.argmin(lenTotal)
	maxLenIndex = np.argmax(lenTotal)
	leftIndex = [0, 1, 2, 3]
	for i in leftIndex:
		if i == minLenIndex or i == maxLenIndex:
			leftIndex.remove(i)
	cLeft1 = c[leftIndex[0]]
	cLeft2 = c[leftIndex[1]]
	if cLeft1[1] > cLeft2[1]:
		cLeft1, cLeft2 = (cLeft2, cLeft1)
	topLeft = np.array((c[minLenIndex][0], c[minLenIndex][1]))
	topRight = np.array((cLeft1[0], cLeft1[1]))
	bottomRight = np.array((c[maxLenIndex][0], c[maxLenIndex][1]))
	bottomLeft = np.array((cLeft2[0], cLeft2[1]))
	w1 = np.sqrt(np.sum((topRight - topLeft) * (topRight - topLeft)))
	w2 = np.sqrt(np.sum((bottomRight - bottomLeft) * (bottomRight - bottomLeft)))
	h1 = np.sqrt(np.sum((bottomLeft - topLeft) * (bottomLeft - topLeft)))
	h2 = np.sqrt(np.sum((bottomRight - topRight) * (bottomRight - topRight)))
	staRatio = w / h
	ratio1 = w1 / h1
	ratio2 = w2 / h2
	if (ratio1 >= staRatio - thresh and ratio1 <= staRatio + thresh) and (ratio2 >= staRatio - thresh and ratio2 <= staRatio + thresh):
		return True, (topLeft, topRight, bottomRight, bottomLeft), getSkewScale(topLeft, topRight, bottomRight, bottomLeft)
	else:
		return False, (), []

def determingCorrectCircles(circles, w, h):
	if len(circles) < 4:
		return []
	for currentCircleIndex, currentCircle in enumerate(circles):
		leftCircles = circles[range(currentCircleIndex + 1, len(circles))].tolist();
		leftCircles.extend(circles[range(0, currentCircleIndex)])
		c1 = currentCircle
		for windowIndex in range(len(leftCircles)):
			c2, c3, c4 = getSlideWindow(leftCircles, windowIndex, 3)
			result, corners, skewScale = determineBoxRatio((c1[0], c1[1]), (c2[0], c2[1]), (c3[0], c3[1]), (c4[0], c4[1]), w, h)
			if result:
				return corners, (c1, c2, c3, c4), skewScale
	return [], (), []

# main function, paperW,paperH: 目标区域宽高, blockList = [(0.3,0.3,0.5,0.5), ]左上右下
def houghTestCircle(originalImg, paperW, paperH, blockList = None, scaleThresh = 0.3):
	imgSize = getImgSize(originalImg)
	w, h = imgSize
	# 按比例缩放
	w, h = (np.int(w * scaleThresh), np.int(h * scaleThresh))
	dw, dh = (int(paperW * scaleThresh * 2), int(paperH * scaleThresh * 2))
	minWH = np.min((w, h))
	originalImg = cv2.resize(originalImg, (w, h))

	img = grayImg(originalImg)
	# 调试， 画圆
	imgColor = originalImg.copy()
	# 调试， 画圆
	imgColor02 = originalImg.copy()
	# 切割结果
	splitArea = np.array([])

	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minWH * 0.3, param1 = 60, param2 = 15, minRadius = 1, maxRadius = 20)
	# 调试，轮廓
	img = cv2.Canny(img, 15, 60, apertureSize = 3)
	# 确定四个边角圆
	# skewScale: 边界框四个角畸变值
	corners, correctCircles, skewScale = determingCorrectCircles(circles[0], paperW, paperH)
	corners = np.array(corners, dtype = np.float32)
	# 未过滤的圆
	print "circles: ", circles
	# 过滤后的圆
	print "correctCircles: ", correctCircles
	# 过滤后的圆的圆心
	print "corners: ", corners
	# 畸变值
	print "skew: ", skewScale
	# 画出过滤前的圆
	if circles.any():
		# 调试：画圆
		circles = np.uint16(np.around(circles))
		for i in circles[0]:
			cv2.circle(imgColor02,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(imgColor02,(i[0],i[1]),2,(0,0,255),3)
	# 画出过滤后的圆
	if correctCircles:
		# 调试：画圆
		correctCircles = np.uint16(np.around(correctCircles))
		for i in correctCircles:
			cv2.circle(imgColor,(i[0],i[1]),i[2],(0,255,0),2)
			cv2.circle(imgColor,(i[0],i[1]),2,(0,0,255),3)
		# 映射，切割
		transPs = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype = np.float32)
		transform = cv2.getPerspectiveTransform(corners, transPs)
		splitArea = cv2.warpPerspective(src = originalImg, M = transform, dsize =  (dw, dh))
	# 调试:切割目标区域
	blockListImg = []
	if blockList:
		for blockCorner in blockList:
			# 待切割区域对角点
			topLeftX, topLeftY, bottomRightX, bottomRightY = np.array(blockCorner) * np.array([dw, dh, dw, dh])
			# 待切割区域宽高
			tmpW = int(bottomRightX - topLeftX)
			tmpH = int(bottomRightY - topLeftY)
			# 待切割区域四个边角, 不抵消畸变
			fourPoints = np.array([[topLeftX, topLeftY], [bottomRightX, topLeftY], [bottomRightX, bottomRightY], [topLeftX, bottomRightY]])
			# 待切割区域四个边角，抵消畸变
			# 左上，右上，右下，左下
			fourPointsAntiSkew = np.array([[topLeftX, topLeftY], [bottomRightX, topLeftY], [bottomRightX, bottomRightY], [topLeftX, bottomRightY]]) - np.array(skewScale)
			
			# 结果图像四个边角
			transPsTmp = np.array([[0, 0], [tmpW, 0], [tmpW, tmpH], [0,tmpH]], dtype = np.float32)

			cornersTmp = np.array(fourPoints, dtype = np.float32)
			cornersTmpSkew = np.array(fourPointsAntiSkew, dtype = np.float32)
			
			transform = cv2.getPerspectiveTransform(cornersTmp, transPsTmp)
			splitAreaTmp = cv2.warpPerspective(src = splitArea, M = transform, dsize =  (tmpW, tmpH))

			transformSkew = cv2.getPerspectiveTransform(cornersTmpSkew, transPsTmp)
			splitAreaTmpSkew = cv2.warpPerspective(src = splitArea, M = transformSkew, dsize =  (tmpW, tmpH))

			blockListImg.extend([splitAreaTmp, splitAreaTmpSkew])
	showImg(img, imgColor02, imgColor, splitArea, *blockListImg)
	return

# main function
def houghTest(img):
	img = cv2.resize(img, (600, 700))
	originalImg = img
	# 增加亮度和对比度, 加高光可以突出黑色区域， 弱化灰色区域
	# img = cv2.convertScaleAbs(img, alpha = 1, beta = -60)
	# 灰度图
	img = grayImg(img)
	imgSize = getImgSize(img)
	minW = min(img.shape)
	maxW = max(img.shape)
	img = cv2.blur(img, (4, 4))
	# 画出轮廓
	edges0 = cv2.Canny(img, 10, 50, apertureSize = 3)
	# edges = sobel(img)
	# edges = cv2.blur(edges, (8, 8))
	# edges = erosion(edges, iterations = 1)
	edges = dilation(edges0, iterations = 2)
	# 调试: 整体轮廓
	edges = cv2.Canny(edges, 10, 50, apertureSize = 3)
	# 寻找边框
	# contours, hierarchy = findContours(edges, cv2.RETR_TREE)
	# 白色外层轮廓
	# blackImg = createBlackImg(img.shape)
	# drawContours(blackImg, contours, (255, 255, 255), 1)
	# drawContours(blackImg, contours, (255, 255, 255), 1)
	# 边框黑色外层轮廓
	# whiteImg = createWhiteImg(imgSize)
	# 调试：绘制边框
	# drawContours(whiteImg, contours, (0, 0, 0), 1)
	# boundingBoxes = getBoundingRect(contours)
	# for boundingBox in boundingBoxes:
	# 	cv2.rectangle(img, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]), (0, 0, 255), 1)
 	# 先找轮廓边线， 再找边线交出的矩形
	# lines = cv2.HoughLines(edges, 1, np.pi / 360, 200)
	# lines = getLinesFromPolarCoord(lines, maxW)
	# lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 50, minLineLength = 100, maxLineGap = 20)
	lines = np.array(cv2.HoughLinesP(edges, 1, np.pi / 360, 50, minLineLength = 100, maxLineGap = 50))
	# lines = remainLine(lines)
	# 求倾角众数
	mode = findMode(findAngle(lines), 7, 9)
	mode[0][1][1].extend(mode[1][1][1])
	lines = lines[mode[0][1][1]]
	# 调试: 画出延长线
	lines0 = expandLine(lines, int(maxW))
	wimg0 = createWhiteImg(getImgSize(img))
	drawLines(wimg0, lines0)
	# 调试：画出边线
	wimg = createWhiteImg(getImgSize(img))
	drawLines(wimg, lines)
	# 计算交点
	intersectPoints = getIntersectPoints(lines, imgSize)
	cornerPoints = getBoundingCornerPoints(intersectPoints, imgSize)
	# 微调, 收缩边框
	thresh = 190.0
	threshZero = 1.0
	cornerPoints[0] += minW / thresh * threshZero
	cornerPoints[1][0] -= minW / thresh * threshZero
	cornerPoints[1][1] += minW / thresh * threshZero
	cornerPoints[2] -= minW / thresh * threshZero
	cornerPoints[3][0] += minW / thresh * threshZero
	cornerPoints[3][1] -= minW / thresh * threshZero
	transPs = np.array([[0, 0], [imgSize[0], 0], [imgSize[0], imgSize[1]], [0, imgSize[1]]], dtype = np.float32)
	transform = cv2.getPerspectiveTransform(cornerPoints, transPs)
	# 调试:画出边框
	wimg2 = createWhiteImg(imgSize)
	drawRectP4(wimg2, np.array(cornerPoints).astype(np.int32))
	# 画出前景
	wimg3 = cv2.warpPerspective(src = originalImg, M = transform, dsize =  imgSize)
	# 在检测到的直线上找边框
	# contours, hierarchy = findContours(wimg, cv2.RETR_TREE)
	# contours, hierarchy = findContours(blackImg, cv2.RETR_TREE)
	# wimg = createWhiteImg(imgSize)
	# 画出边框
	# drawContours(wimg, contours, (0, 0, 0), 1)
	# 找到面积最大的区域
	# maxAreaContours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
	# 求出其最小包围矩形
	# rect = cv2.minAreaRect(maxAreaContours)
	# 调试：画出矩形
	# box = np.int0(cv2.boxPoints(rect))
	# whiteImg = createWhiteImg(imgSize)
	# drawContours(wimg, [maxAreaContours], (0, 0, 0), 1)
	showImg(wimg0, wimg, wimg2, edges0, edges, originalImg, img, wimg3)

# main function
def ansAreaSplit(img):
	img = cv2.resize(img, (600, 700))
	originalImg = img
	img = grayImg(img)
	imgSize = getImgSize(img)
	minW = min(img.shape)
	maxW = max(img.shape)
	img = cv2.blur(img, (4, 4))
	# 画出轮廓
	edges0 = cv2.Canny(img, 10, 50, apertureSize = 3)
	edges = dilation(edges0, iterations = 2)
	#edges = erosion(edges, iterations = 1)
	#edges = erosion(edges, iterations = 2)
	#edges = dilation(edges0, iterations = 3)
	# 边框
	contours, hierarchy = findContours(edges, cv2.RETR_TREE)
	blackImg = createBlackImg(imgSize)
	drawContours(blackImg, contours, (255, 255, 255), 1)

	showImg(edges0, edges, blackImg)

# 从填涂区域包围框中选出正确的填涂框, 并计算中心坐标
# hierarchy结构：(前一个框序号，后一个框序号，第一个子框序号，父框序号)
# 序号为-1表示无
# 返回所有填涂区域的中心坐标和区域框
def findAnswerBoxCenter(rectangles, hierarchy):
	hierarchy = hierarchy[0]
	ansBoxCenter = []
	topBoundingBox = []
	for index, item in enumerate(hierarchy):
		if item[3] != -1:
			x1, y1, w, h = rectangles[index]
			# centerX， centerY，S
			ansBoxCenter.append(((x1 + x1 + w) / 2, (y1 + y1 + h) / 2, w * h))
		elif rectangles[index][2] != -1:
			topBoundingBox = rectangles[index]
	return ansBoxCenter, topBoundingBox

# 通过填涂区域坐标确定填涂的答案
# ansBoxCenter为填涂区域中心坐标
# questionCount为竖排的题目数
# answerCount为每个题目备选答案数目
# W,H为作答区域的宽高
# 返回答题结果，questionCount * answerCount的二位数组，置1位代表填涂，置0位代表未填涂
def determineAnswer(ansBoxCenter, questionCount, answerCount, W, H, restrictArea = True, restrictAreaThresh = 0.2):
	baseX = baseY = 0
	stepX = int(float(W) / questionCount)
	stepY = int(float(H) / answerCount)
	standardS = stepX * stepY
	answerMap = np.zeros((questionCount, answerCount))
	for anSenter in ansBoxCenter:
		ansX, ansY, S = anSenter
		# 填涂面积至少为判定方格面积的20%
		if restrictArea and S <= restrictAreaThresh * standardS:
			continue
		ansIndex = int((ansY - baseY) / stepY)
		quesIndex = int((ansX - baseX) / stepX)
		answerMap[quesIndex][ansIndex] = 1
	return answerMap

# 横跨整个答题卡的长条形作答区域，包含多个题组
# 通过填涂区域坐标确定填涂的答案
# ansBoxCenter为填涂区域中心坐标
# questionCount为[!每个!]题组的竖排的题目数
# answerCount为每个[!每个!]题组的题目备选答案数目
# groupCount为题组个数
# W,H为作答区域的宽高
# spaceStep为每个题组的间隔长度，默认一个填涂区域大小
# 返回答题结果，questionCount * answerCount的二位数组，置1位代表填涂，置0位代表未填涂
def determineAnswerBar(ansBoxCenter, questionCount, answerCount, groupCount, W, H, spaceStep = 1, restrictArea = True, restrictAreaThresh = 0.2):
	baseX = baseY = 0
	stepX = int(float(W) / (questionCount * groupCount + groupCount - 1))
	stepY = int(float(H) / answerCount)
	standardS = stepX * stepY
	answerMap = np.zeros((questionCount * groupCount, answerCount))
	for anSenter in ansBoxCenter:
		ansX, ansY, S = anSenter
		# 填涂面积至少为判定方格面积的20%
		if restrictArea and S <= restrictAreaThresh * standardS:
			continue
		ansIndex = int((ansY - baseY) / stepY)
		quesIndePre = int((ansX - baseX) / stepX)
		# gapIndex 填涂区域所属组序号， 从1编号
		# determine spacegap index
		# quesIndex -> [q * (i - 1) + 1, (q + 1) * i - 1], where q:questionCount, i:gapIndex
		# q * (i - 1) + 1 <= x and (q + 1) * i - 1 >= x, where x: quesIndex
		# resolve this euqation
		# i -> [(x + 1) / (q + 1), (x + q) / (q + 1)]
		gapIndex = np.ceil(float((quesIndePre + 1)) / (questionCount + 1)) # or np.floor(float((quesIndePre + questionCount)) / (questionCount + 1))
		quesIndex = int(quesIndePre - gapIndex + 1)
		answerMap[quesIndex][ansIndex] = 1
	return answerMap

# main function
def readCard(img):
	questionCount = 5
	answerCount = 5
	# 增加亮度和对比度, 加高光可以突出黑色区域， 弱化灰色区域
	# img = cv2.convertScaleAbs(img, alpha = 1.3, beta = 20)
	# 确定裁剪范围，row1,row2,col1,col2
	areas = [[245, 325, 320, 480], [258, 325, 485, 615], [245, 325, 10, 615]]
	area = None#areas[1]
	groupCount = 1
	# 灰度图
	originalImg = img
	img = grayImg(img)
	w, h = img.shape
	imgSize = getImgSize(img)
	# otsu二值化
	# img = binaryInv(img)
	# 低通滤波
	# img = cv2.blur(img, (3, 3))
	# 腐蚀, 实际效果为涂抹填涂区域
	# img = erosion(img)
	# 膨胀， 实际效果为缩小填涂区域
	# img = dilation(img, iterations = 1)
	# 裁剪
	# 裁剪
	if area:
		rectImg01 = img[area[0]: area[1], area[2]: area[3]]
	else:
		rectImg01 = img
	# 角点检
	# rectImg01_32 = np.float32(rectImg01)
	# rectImg01 = cv2.cornerHarris(np.float32(rectImg01_32), 2, 3, 0.04)
	# rectImg01 = dilation(rectImg01)
	# rectImg01 = erosion(rectImg01)
	# rectImg01 = binaryThresh(rectImg01, 0.01 * rectImg01.max(), 255, 1)
	# # find centroids
	# rectImg01 = np.uint8(rectImg01)
	# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(rectImg01)
	# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	# corners = np.int0(cv2.cornerSubPix(rectImg01_32, np.float32(centroids), (5, 5), (-1, -1), criteria))
	# 寻找填涂区域数量
	rectImg010 = erosion(rectImg01, iterations=3)
	contours, hierarchy = findContours(rectImg010, cv2.RETR_TREE)
	whiteImg00 = createWhiteImg(imgSize)
	drawContours(whiteImg00, contours, (0, 0, 0), 1)
	print "Count: ", len(hierarchy[0]) - 1
	ansCount = len(hierarchy[0]) - 1

	corners = np.int0(cv2.goodFeaturesToTrack(rectImg01, 20, 0.01, 10))
	# 聚类，找出角点中心
	cornersStack = np.hstack((corners[:, :, 0], corners[:, :, 1]))
	pred = KMeans(n_clusters = ansCount).fit_predict(cornersStack)
	_cornerMap = {}
	for index, _corner in enumerate(cornersStack):
		if pred[index] not in _cornerMap:
			_cornerMap[pred[index]] = []
		_cornerMap[pred[index]].append(_corner)
	newCorners = []
	for v in _cornerMap.values():
		_sum = np.zeros(2)
		c = 0
		for p in v:
			_sum += p
			c += 1
		avg = _sum / c
		newCorners.append(np.int32(avg))
	newCorners = np.array(newCorners)
	# print corners
	# originalImg[corners[:, 0, 1], corners[:, 0, 0]] = [0, 0, 255]
	# rectImg01 = binaryThresh(rectImg01, 50, 255, cv2.THRESH_BINARY_INV)
	# rectImg01 = np.uint8(rectImg01)
	# # 白边  
	# row, col = rectImg01.shape
	# rectImg01[0] = 255
	# rectImg01[row - 1] = 255
	# rectImg01[:, col - 1] = 255
	# rectImg01[:, 0] = 255
	# #rectImg01[1] = 255
	# # 调试:寻找并在白色底图上画出轮廓
	originalImg[newCorners[:, 1], newCorners[:, 0]] = [0, 0, 255]
	# # 找出轮廓
	# 角点标注
	whiteImg = createWhiteImg(imgSize)
	whiteImg[newCorners[:, 1], newCorners[:, 0]] = 0
	contours, hierarchy = findContours(whiteImg, cv2.RETR_TREE)
	# # 调试:画出轮廓
	drawContours(whiteImg, contours, (0, 0, 0), 1)
	# 得到填涂答案
	boundingBox = getBoundingRect(contours)
	ansBoxCenter, topBoundingBox = findAnswerBoxCenter(boundingBox, hierarchy)
	# 单个题组
	# ansMap = determineAnswer(ansBoxCenter, 5, 4, topBoundingBox[2] - topBoundingBox[0], topBoundingBox[3] - topBoundingBox[1])
	# 四个题组
	# ansMap = determineAnswerBar(ansBoxCenter, 5, 5, groupCount, topBoundingBox[2] - topBoundingBox[0], topBoundingBox[3] - topBoundingBox[1]
	# 	, restrictArea = True, restrictAreaThresh = 0.05)
	ansMap = determineAnswerBar(ansBoxCenter, questionCount, answerCount, groupCount, topBoundingBox[2] - topBoundingBox[0], topBoundingBox[3] - topBoundingBox[1]
		, restrictArea = True, restrictAreaThresh = 0.00)
	# 调试:画出轮廓
	showImg(whiteImg00, rectImg01, whiteImg)	
	# showImg(whiteImg)	
	return ansMap.T

if __name__ == "__main__":
	img = readImg("/Volumes/SD/ML/scantron/pics/card01.jpg")
	#houghTest(img)
	readCard(img)
	