from detector import *
import os

def main():
    video = 0

    config = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    model = os.path.join("model_data", "frozen_inference_graph.pb")
    classes = os.path.join("model_data", "coco.names")

    detector = Detector(video, config, model, classes)
    detector.onVideo()

if __name__ == '__main__':
    main()