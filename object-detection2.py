import torch
import numpy as np
import cv2
from time import time

class ObjectDetection:

    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt',force_reload=True)        
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        return model
    
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame) 
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
       
        return labels, cord
    
    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]
    
    def make_CSV(self,results,framecount):
        file = open("annotation.csv",'a')
        
        labels, cord = results
        n=len(labels)
        # print(n)
        for i in range(n):
            class_ID = self.class_to_label(labels[i])
            str_result = str(cord[i])
            file.write(str_result+class_ID)
            file.write("\n")
        file.write(str(framecount))
        file.write("\n")
        file.close()


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        # class_list = ["car","motorcycle","bus","truck"]
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            class_ID = self.class_to_label(labels[i])
            row = cord[i]
            # if row[4] >= 0.5 and class_ID in class_list:
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


    def __call__(self):
        cap = cv2.VideoCapture("input/test6.mp4")
    
        framecount = 0
        while(cap.isOpened()):
            ret, video_frame=cap.read()
            video_frame = cv2.resize(video_frame, (1000,640))
            
            
            if ret==True:
                framecount=framecount+1 
                results = self.score_frame(video_frame)
                video_frame = self.plot_boxes(results, video_frame)
                
                self.make_CSV(results,framecount)
                cv2.imshow("Source", video_frame)
            key = cv2.waitKey(30)
            # if key q is pressed then break 
            if key == 113:
                break 
        cap.release()
        cv2.destroyAllWindows()



detection = ObjectDetection()
detection()
