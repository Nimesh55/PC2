import enum
import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
needed_classes = []

with open('coco.names','r') as f:
    classes = f.read().splitlines()
with open('coco_modified.names','r') as f:
    needed_classes = f.read().splitlines()

cap =cv2.VideoCapture('test03.mp4')

# Setup initial positions in the line
_, img = cap.read()
height, width,_ = img.shape
line_position = 3*height//4
line_left = 90
line_left_down = 25
line_right = width-90
line_right_down = width-25

while True:
    _, img = cap.read()
    height, width,_ = img.shape

    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True, crop=False)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes),3))

    cv2.line(img, (line_left, line_position),(line_right, line_position),(255,127,0), 4)
    cv2.line(img, (line_left, line_position),(line_left_down, line_position+25),(255,127,0), 4)
    cv2.line(img, (line_right, line_position),(line_right_down, line_position+25),(255,127,0), 4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            if label not in needed_classes:
                continue
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img, label, (x,y+h), font, 2, color, 2)

            if(y<line_position and y+h>line_position):
                textSize = cv2.getTextSize("Slow down vehicle",font,3,3)
                x0 = width//2-textSize[0][0]//2
                y0 = height-30
                cv2.rectangle(img, (x0-5,y0+5), (x0+textSize[0][0],y0-textSize[0][1]-5), (255,255,255), -1)
                cv2.putText(img, "Slow down vehicle", (x0, y0),font,3,(255,0,0),3)


    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 119: #up
        if line_position>0:
            line_position -= 5
    elif key == 115: #down
        if line_position<height:
            line_position += 5
    elif key == 97: #left
        line_left-=5
    elif key == 65: #left reverse
        line_left+=5
    elif key == 122: #left down
        line_left_down-=5
    elif key == 90: #left down reverse
        line_left_down+=5
    elif key == 100: #right
        line_right+=5
    elif key == 68: #right reverse
        line_right-=5
    elif key == 99: #right down
        line_right_down+=5
    elif key == 67: #right down reverse
        line_right_down-=5

cap.release()
cv2.destroyAllWindows()