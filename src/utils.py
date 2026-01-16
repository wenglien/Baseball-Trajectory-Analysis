import cv2
import copy
import numpy as np
from src.FrameInfo import FrameInfo


def draw_ball_curve(frame, trajectory, max_points: int | None = 15):
    """
    在畫面上畫出球的軌跡：
      - 只顯示軌跡最後 max_points 個點，讓線條長度與球速比較貼合，
        避免一開始就出現貫穿整個畫面的長線。
    """
    trajectory_weight = 0.5
    temp_frame = frame.copy()

    if len(trajectory):
        # Only take the last max_points points, so that the trajectory length is fixed in a small "tail"
        if max_points is not None and len(trajectory) > max_points:
            traj = trajectory[-max_points:]
        else:
            traj = trajectory

        ball_points = copy.deepcopy(traj)
        for point in ball_points:
            color = point[2]
            del point[2]
        ball_points = np.array(ball_points, dtype="int32")
        # Thinner the line, so that the trajectory is closer to the actual flight path and feel
        cv2.polylines(temp_frame, [ball_points], False, color, 10, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(
            temp_frame, trajectory_weight, frame, 1 - trajectory_weight, 0
        )
        last_ball = tuple(traj[-1][:-1])
        # Make the end ball smaller, so that it does not cover too much of the screen
        cv2.circle(frame, tuple(last_ball), 8, (255, 255, 255), -1)
    return frame


def fill_lost_tracking(frame_list):
    balls_x = [frame.ball[0] for frame in frame_list if frame.ball_in_frame]
    balls_y = [frame.ball[1] for frame in frame_list if frame.ball_in_frame]

    # print(balls_x)
    # print(balls_y)

    # If there are no or almost no ball detection points, skip the filling, avoid polyfit error
    # At least three points are needed to fit the quadratic polynomial
    if len(balls_x) < 3:
        return

    # Get the polynomial equation
    curve = np.polyfit(balls_x, balls_y, 2)
    poly = np.poly1d(curve)

    lost_sections = []
    in_lost = False
    frame_count = 0

    # Get the sections where the ball is lost tracked
    for idx, frame in enumerate(frame_list):
        if frame.ball_lost_tracking and frame_count == 0:
            in_lost = True
            lost_sections.append([])

        if in_lost and not (frame.ball_lost_tracking):
            in_lost = False
            frame_count = 0

        if in_lost:
            lost_sections[-1].append(idx)
            frame_count += 1

    # Modify the frames in lost section with the approximated ball
    for lost_section in lost_sections:
        if lost_section:
            prev_frame = frame_list[lost_section[0] - 1]
            last_frame = frame_list[lost_section[-1] + 1]
            color = prev_frame.ball_color

            lost_idx = [frame_list[i] for i in lost_section]

            # Speed is the x difference for each frame
            diff = last_frame.ball[0] - prev_frame.ball[0]
            speed = int(diff / (len(lost_idx) + 1))

            for idx, frame in enumerate(lost_idx):
                x = prev_frame.ball[0] + (speed * (idx + 1))
                y = int(poly(x))
                frame.ball_in_frame = True
                frame.ball = (x, y)
                frame.ball_color = color
                # print('Fill', x, y)


def distance(x, y):
    temp = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    return temp ** (0.5)
