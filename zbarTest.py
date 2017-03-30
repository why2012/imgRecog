import zbar
from PIL import Image

def scanImgPickOne(path):
	resultCode = ""
	img = Image.open(path).convert('L')
	w, h = img.size
	zbarImg = zbar.Image(w, h, 'Y800', img.tostring())#img.tobytes()
	print img.size
	scanner = zbar.ImageScanner()
	barCodeCount = scanner.scan(zbarImg)
	for scanResult in zbarImg:
		resultCode = scanResult.data
		break
	del zbarImg
	return resultCode