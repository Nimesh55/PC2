import enum
import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []

with open('coco.names','r') as f:
    classes = f.read().splitlines()

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
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img, label, (x+w//2,y+h//2), font, 2, color, 3)

            if(y<line_position and y+h>line_position):
                cv2.putText(img, "Slow down vehicle", (width//6, height-10),font,3,(255,0,0),3)

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