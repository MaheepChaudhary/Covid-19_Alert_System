#importing essential packages
import cv2 as cv
import numpy as np
import os

#importing the trianed model neural net
net = cv.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")

classes = []

with open("coco.names") as f:
  classes = [line.strip() for line in f.readlines()]
print(classes)

name = net.getLayerNames()
print(name)

out = net.getUnconnectedOutLayers()
print(out)

outputlayernames = [name[i[0]-1] for i in out]
print(outputlayernames)

cap = cv.VideoCapture("test_video.mp4")
_,frame = cap.read()
out = cv.VideoWriter('output.mp4',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
while(True):
  ret,img = cap.read()  
  (H,W) =img.shape[:2]
  Green = [int(0),int(255),int(0)]
  blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
  net.setInput(blob)
  output = net.forward(outputlayernames)

  boxes,confidences,class_ids = [],[],[]
  height,width,shape = img.shape

  for o in output:
    for detection in o:
      scores = detection[5:]
      class_ids = np.argmax(scores)
      confidence = scores[class_ids]
      if confidence>0.6 and class_ids == 0:
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))

  font = cv.FONT_HERSHEY_PLAIN
  indexes = cv.dnn.NMSBoxes(boxes,confidences,0.5,0.5)

  rb,red_boxes = [],[]

  for i in range(len(boxes)):
    if i in indexes:
      a,b,c,d = boxes[i]
      cv.rectangle(img,(int(a),int(b)),(int(c+a),int(d+b)),(0,255,0),2)
  #    cv.putText(img,str(confidences[i]),(int(a),int(b)+30),font,0.5,[0,255,0],2)
      rb.append([a,b,c,d])

  print(len(rb))

  for f in range(0,len(rb)):
    for j in range(0,len(rb)):
      if j == f:
        break
      else:  
        x1,y1 = rb[f][:2]
        x2,y2 = rb[j][:2]
        dist = np.sqrt(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1)))
        if dist<=120:
          red_boxes.append(rb[f])
          red_boxes.append(rb[j])

  for r in range(len(red_boxes)):
    A,B,C,D = red_boxes[r]
    cv.rectangle(img,(A,B),(A+C,B+D),[0,0,255],2)
    cv.putText(img,"ALERT",(int(A),int(B)-5),font,0.5,[0,0,255],2)    

  if cv.waitKey(1) and 0xFF == ord("q"):
    break
                
  out.write(img)              

out.release()
cap.release()
cv2.destroyAllWindows()  