import cv2
import time


def run():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    result, image = cam.read()
    i = 0
    cv2.namedWindow("Image1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image1", 875, 650)
    cv2.moveWindow("Image1", 400, 70)
    while result:
        cv2.imshow("Image1", image)
        result, image = cam.read()
        key = cv2.waitKey(20)
        if key == 32:
            cv2.imwrite(f"Dude{i}.jpg", image)
            i += 1
        if key == 27:
            break
    cam.release()
    cv2.destroyWindow("Image1")


def generate_camera_feed():
    camera = cv2.VideoCapture(0)
    while cv2.waitKey(20) != "32":
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



