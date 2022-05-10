from flask import Flask, render_template, Response
import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize





app = Flask(__name__)
# use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    estimator = load_pretrain_model('VGG_origin')
    action_classifier = load_action_premodel('Action/framewise_recognition_under_scene.h5')


    realtime_fps = '0.0000'
    start_time = time.time()
    fps_interval = 1
    fps_count = 0
    run_timer = 0
    frame_count = 0

    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()

    camera = choose_run_mode(args)
    
    while True:
        # Capture frame-by-frame
        success, show = camera.read()  # read the camera frame
        if not success:
            break
        else:
            fps_count += 1
            frame_count += 1

            # pose estimation
            humans = estimator.inference(show)
            # get pose info
            pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
            # recognize the action framewise
            show = framewise_recognize(pose, action_classifier)

            height, width = show.shape[:2]

            if (time.time() - start_time) > fps_interval:
                realtime_fps = fps_count / (time.time() - start_time)
                fps_count = 0  
                start_time = time.time()
            fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
            cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            num_label = "Human: {0}".format(len(humans))
            cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if frame_count == 1:
                run_timer = time.time()
            run_time = time.time() - run_timer
            time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
            cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            ret, buffer = cv.imencode('.jpg', show)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    # global estimator , action_classifier, realtime_fps, start_time, fps_interval, fps_count, run_timer, frame_count, camera, video_writer
    app.run(debug=True)