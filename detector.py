import cv2

import numpy as np


class Detector:
    def __init__(self, video, config, model, classes):
        self.video = video
        self.config = config
        self.model = model
        self.classes = classes


        self.net = cv2.dnn.DetectionModel(self.model, self.config)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classes, 'r') as f:
            self.classeslist = f.read().splitlines()
        
        self.classeslist.insert(0, '__Background__')

        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classeslist), 3))

        
        print(self.classeslist)

    def onVideo(self):
        cap = cv2.VideoCapture(self.video)

        if (cap.isOpened() == False):
            print("error")
            return
        
        (success, image) = cap.read()

        while success:
            classLableIDs, confidence, bboxs = self.net.detect(image, confThreshold = 0.4)
            
            bboxs = list(bboxs)
            confidence = list(np.array(confidence).reshape(1,-1)[0])
            confidence = list(map(float, confidence))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidence, score_threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidence[np.squeeze(bboxIdx[i])]
                    classLableID = np.squeeze(classLableIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classeslist[classLableID]
                    classColor = [int(c) for c in self.colorList[classLableID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x,y,w,h = bbox

                    cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,255,255), thickness=1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            cv2.imshow("Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()
        
