from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
from functools import partial
from skimage import color
from skimage import io
import numpy as np

width = 280
height = 280
center = height/2
white = (255, 255, 255)
black = (0,0,0)

#----------Convolutional Neural Network-------------------

def getAns(m,w):
    
    from keras.models import load_model
    if w.get() == 1:
        model = load_model('realmodel.h5')
    elif w.get() == 2:
        model = load_model('realmodel2.h5')
    elif w.get() == 3:
        model = load_model('realmodel3.h5')
    else:
        model = load_model('realmodel4.h5')
    
    '''from PIL import Image
    import numpy as np
    
    img = Image.open('image1.png').convert('LA')
    img.load()
    img.show()'''
    
    img1 = color.rgb2gray(io.imread('image1.png'))
    
    #img = np.asarray( img, dtype="uint8")
    #img = img.reshape((1, 784)).astype('float32')
    img1 = img1.reshape((1,28,28,1)).astype('float32')
    
    ans = model.predict(img1)
    max_prob = 0
    my_ans = 0
    cnt = 0
    for i in np.nditer(ans):
        if i > max_prob:
            max_prob = i
            my_ans = cnt
        cnt = cnt + 1
    m.config(text=my_ans)
    

#---------------------------------------------------------

#--------------Processing the Image-----------------------

def process():
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

#---------------------------------------------------------

def save():
    filename = "image.png"
    image1.save(filename)
    process()

def clear(canvas, m):
    canvas.delete("all")
    draw.rectangle([(0,0),(280,280)],fill="black",outline="black")
    filename = "image.png"
    image1.save(filename)
    m.config(text='')

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="white",width=20)
    draw.line([x1, y1, x2, y2],fill="white",width=20)

root = Tk()
root.title('Digit Recognizer')
root.resizable(0,0)

cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

image1 = PIL.Image.new("RGB", (width, height), black)
draw = ImageDraw.Draw(image1)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

ans = Label(text='')
w = Scale(from_=1, to=4, orient=HORIZONTAL)
sv_button=Button(text="Save",command=save)
dec_button=Button(text="Detect",command=partial(getAns, ans, w))
clr_button=Button(text="Clear",command=partial(clear, cv, ans))
ans.pack()
sv_button.pack()
dec_button.pack()
clr_button.pack()
w.pack()

root.mainloop()