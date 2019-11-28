import numpy as np
    import cv2
    from scipy import ndimage

    def getBestShift(img):
        cy, cx = ndimage.measurements.center_of_mass(img)
    
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)
    
        return shiftx, shifty
    
    
    def shift(img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    image = cv2.imread("image.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    
    gray = cv2.resize(gray, (20, 20))
    
    colsPadding = 4, 4
    rowsPadding = 4, 4
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
    
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted
    
    cv2.imwrite("image1.png", gray)