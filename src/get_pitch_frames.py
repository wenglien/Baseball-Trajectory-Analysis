import colorsys
import copy
import random
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import shift

from src.FrameInfo import FrameInfo
from src.utils import distance, fill_lost_tracking
from src.generate_overlay import generate_overlay, draw_ball_curve
from src.SORT_tracker.sort import Sort
from src.SORT_tracker.tracker import Tracker

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Get the pitching section in the whole video
def get_pitch_frames(
    video_path, infer, input_size, iou, score_threshold, show_preview: bool = True
):
    print("Video from: ", video_path)
    vid = cv2.VideoCapture(video_path)
    
    # Check if the video is successfully opened
    if not vid.isOpened():
        raise ValueError(f"無法開啟影片檔案：{video_path}\n請確認檔案格式是否支援（mp4/avi/mov/mkv）。")

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    # Check if the video parameters are valid
    if width <= 0 or height <= 0:
        vid.release()
        raise ValueError(f"無法讀取影片尺寸，可能是檔案損壞或格式不支援：{video_path}")
    if fps <= 0:
        fps = 30 
        print(f"警告：無法讀取 fps，使用預設值 30")

    track_colors = [
        (161, 235, 52),
        (83, 254, 92),
        (255, 112, 52),
        (161, 235, 52),
        (255, 235, 52),
        (255, 38, 38),
        (255, 235, 52),
        (210, 235, 52),
        (52, 235, 131),
        (52, 64, 235),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 127),
        (127, 0, 127),
        (255, 127, 255),
        (127, 0, 255),
        (255, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (50, 100, 150),
        (10, 50, 150),
        (120, 20, 220),
    ]

    # Store the pitching section in pitch_frames
    pitch_frames = []
    detected_balls = []
    tracked_balls = []
    frames = []
    # The original program uses min_hits=3, which may be too strict for a single pitching video, so it is changed to 1 to make it easier to form a trajectory
    tracker_min_hits = 1
    frame_id = 0

    # Relax max_age, so that the tracking can tolerate short periods of detection failure
    tracker = Sort(max_age=15, min_hits=tracker_min_hits, iou_threshold=0.1)

    # Mediapipe Pose (reuse for whole video)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw the pitching pose skeleton on each frame
            results = pose.process(frame)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            frames.append(FrameInfo(frame, False))
        else:
            print("Processing complete")
            break

        # Detect the baseball in the frame
        detections = detect(
            infer, frame, input_size, iou, score_threshold, detected_balls
        )

        # Feed in detections to obtain SORT tracking
        if len(detections) > 0:
            trackings = tracker.update(np.array(detections))
        else:
            trackings = tracker.update()

        # Add the valid trackings to balls_list
        for t in trackings:
            t = [int(i) for i in t]
            start = (t[0], t[1])
            end = (t[2], t[3])
            # cv2.rectangle(frame, start, end, (255, 0, 0), 5)
            # cv2.putText(frame, str(t[4]), start, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)

            color = track_colors[t[4] % 12]
            centerX = int((t[0] + t[2]) / 2)
            centerY = int((t[1] + t[3]) / 2)
            tracked_balls.append([centerX, centerY, color])

        # Store the frames with ball tracked
        if len(trackings) > 0:
            # Only run at the first track from SORT
            if len(pitch_frames) == 0:
                last_tracked_frame = frame_id
                add_balls_before_SORT(
                    frames, detected_balls, tracked_balls, tracker_min_hits
                )
                # Add prior 20 frames before the first balsadl
                pitch_frames.extend(frames[-20:])

            # Add lost frames if any
            add_lost_frames(frame_id, last_tracked_frame, frames, pitch_frames)

            # Append the frame with detected ball location
            last_ball = tuple(tracked_balls[-1][:-1])
            pitch_frames.append(FrameInfo(frame, True, last_ball, color))
            last_tracked_frame = frame_id

        if show_preview:
            try:
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                detection = cv2.resize((result), (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("result", detection)
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
            except Exception as e:
                print(f"無法顯示預覽視窗（可忽略）：{e}")

        frame_id += 1

    vid.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    # Use Polyfit to approximate the untracked balls
    fill_lost_tracking(pitch_frames)

    # Add five more frames after the last tracked frame
    if len(pitch_frames) > 0:
        pitch_frames.extend(frames[last_tracked_frame : last_tracked_frame + 10])
    return pitch_frames, width, height, fps


# Tensorflow Object Detection API Sample
def detect(infer, frame, input_size, iou, score_threshold, detected_balls):
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.0
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score_threshold,
    )

    boxes = boxes.numpy()
    scores = scores.numpy()
    classes = classes.numpy()
    valid_detections = valid_detections.numpy()

    offset = 100
    # The original threshold is 0.95, which may be too strict for your own video, so it is changed to 0.7 to increase the detection chance
    accuracyThreshold = 0.7
    frame_h, frame_w, _ = frame.shape
    detections = []

    for i in range(valid_detections[0]):
        score = scores[0][i]
        if score > accuracyThreshold:
            coor = boxes[0][i]
            coor[0] = coor[0] * frame_h
            coor[2] = coor[2] * frame_h
            coor[1] = coor[1] * frame_w
            coor[3] = coor[3] * frame_w

            centerX = int((coor[1] + coor[3]) / 2)
            centerY = int((coor[0] + coor[2]) / 2)

            print(
                f"Baseball Detected ({centerX}, {centerY}), Confidence: {str(round(score, 2))}"
            )
            # cv2.circle(frame, (centerX, centerY), 15, (255, 0, 0), -1)
            detected_balls.append([centerX, centerY])
            detections.append(
                np.array(
                    [
                        coor[1] - offset,
                        coor[0] - offset,
                        coor[3] + offset,
                        coor[2] + offset,
                        score,
                    ]
                )
            )

    return detections


def add_balls_before_SORT(frames, detected, tracked, tracker_min_hits):
    distance_threshold = 100
    first_ball = tracked[0]
    color = first_ball[2]
    balls_to_add = []

    # Get the untracked balls that's close enough to the first tracked ball
    for untracked in detected[-(tracker_min_hits + 1) :]:
        if distance(untracked, first_ball) < distance_threshold:
            untracked.append(color)
            balls_to_add.append(untracked)

    # Add the untracked balls to frame
    modify_frames = frames[-(tracker_min_hits + 1) :]
    balls_to_add_temp = copy.deepcopy(balls_to_add)

    for point in balls_to_add_temp:
        del point[2]
    balls_to_add_temp = np.array(balls_to_add_temp, dtype="int32")

    for idx, frame in enumerate(modify_frames):
        # cv2.polylines(frame.frame, [balls_to_add_temp[:idx+1]], False, color, 22, lineType=cv2.LINE_AA)
        frames[-((tracker_min_hits + 1) - idx)] = FrameInfo(
            frame.frame, True, tuple(balls_to_add_temp[idx]), color
        )


def add_lost_frames(frame_id, last_tracked_frame, frames, pitch_frames):
    if frame_id - last_tracked_frame > 1:
        print("Lost frames:", frame_id - last_tracked_frame)
        frames_to_add = frames[last_tracked_frame:frame_id]

        # Mark the detection in lost in this frame
        for ball_frame in frames_to_add:
            ball_frame.ball_lost_tracking = True
        pitch_frames.extend(frames_to_add)


# def get_bright_color():
#     h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
#     r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
#     return [r, g, b]
