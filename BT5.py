import cv2 as cv
import numpy as np


#Insert small image into another image function
def IMGinsert(im,dog):
    HSVimg = cv.cvtColor(im, cv.COLOR_BGR2HSV) 
    mask = cv.inRange(HSVimg,(0,0,0),(25,195,190))
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(c)
    if w < h:
      scale_val = w
      padding_val = (h - w)//2
      dog = cv.resize(dog, (scale_val,scale_val), interpolation = cv.INTER_CUBIC)
      im[y+padding_val:y+padding_val+scale_val,x:x+scale_val] = dog
    else:
      scale_val = h
      padding_val = (w - h)//2
      dog = cv.resize(dog, (scale_val,scale_val), interpolation = cv.INTER_CUBIC)
      im[y:y+scale_val,x+padding_val:x+padding_val+scale_val] = dog
    cv.rectangle(im, (x, y), (x + w, y + h), (0, 255,0), 2)
    return im


#Scale an image but not break aspect ratio with padding
def padding(img,w,h):
  if w < h:
    tmp = np.zeros(((h-w)//2,w,3))
    img = cv.resize(img, (w,w), interpolation = cv.INTER_CUBIC)
    img = np.concatenate((tmp,img), axis=0)
    img = np.concatenate((img,tmp), axis=0)
  else:
    tmp = np.zeros(h,(w-h)//2,3)
    img = cv.resize(img, (w,w), interpolation = cv.INTER_CUBIC)
    img = np.concatenate((tmp,img), axis=1)
    img = np.concatenate((img,tmp), axis=1)
  return img



  


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv.VideoCapture('test1.mp4')
#get width and height of save video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

dog = cv.imread('dog.jpg')
out = cv.VideoWriter('edit_vid.mp4',cv.VideoWriter_fourcc(*'mp4v'), 29.97, (frame_width,frame_height))

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    frame = IMGinsert(frame,dog)
    cv.imshow('Frame',frame)
    # Write vid
    out.write(frame)

    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      cv.imwrite('record.jpg',frame)
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv.destroyAllWindows()