from flask import Flask, render_template, Response, jsonify
from model import generate_frames, start_camera, stop_camera, get_counts

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start")
def start():
    start_camera()
    return "Camera Started"

@app.route("/stop")
def stop():
    stop_camera()
    return "Camera Stopped"

@app.route("/counts")
def counts():
    return jsonify(get_counts())

if __name__ == "__main__":
    app.run(debug=True)