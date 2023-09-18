import cv2
from flask import *
from get_pic import run, generate_camera_feed
from multiprocessing import Process

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    get_vid = Process(target=web_cam)
    get_vid.start()
    return render_template("index.html")


@app.route("/web_cam")
def web_cam():
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


@app.route("/admin")
def admin():
    return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
