from flask import Flask, render_template, Response, request
from streamer import Streamer
from processor import process_video_file
import requests, json, cv2

app = Flask(__name__)

videos = {
  '1': "/static/sample.mp4",
  '2': "/static/unacademy.mp4",
  '3': "/static/green.MOV",
  '4': "/static/green.mp4",
  '5': "/static/output1.mp4"
}

def setup():
  pass


def gen():
  streamer = Streamer('localhost', 8090)
  streamer.start()

  while True:
    if streamer.client_connected():
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + streamer.get_jpeg() + b'\r\n\r\n')


def gen_processed():
  streamer = Streamer('localhost', 8090)
  streamer.start()

  while True:
    if streamer.client_connected():
      img = streamer.get_frame()
      p_img = process_video_file.remove_green_3(img)
      _, jpeg = cv2.imencode('.jpg', p_img)
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
  return render_template('index.html')


@app.route('/show-video/<path>')
def show(path):
  return Response(process_video_file.resize_video(videos[path]), mimetype='multipart/x-mixed-replace; boundary=frame')
  # return render_template('file_video.html', path=videos[path])
  # return send_from_directory('/static', videos[path], conditional=True)


@app.route('/file-feed/')
def file_feed():
  path = request.args.get("path")
  x = process_video_file.resize_video_1(path)
  return render_template('file_video.html', path=x)


@app.route('/file-feed-stream/')
def file_feed_stream():
  path = request.args.get("path")
  return Response(process_video_file.resize_video_3(path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/multi-stream')
def file_feed_stream_multi():
  path = request.args.get("path")
  rotate = request.args.get("ro")
  return Response(process_video_file.multi_stream_process(path, int(rotate)), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video-feed')
def video_feed():
  return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video-feed-js')
def video_feed_js():
  return Response(gen_processed(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
  app.run(host='localhost', threaded=True, port=5007, debug=False)
