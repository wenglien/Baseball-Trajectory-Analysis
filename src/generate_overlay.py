import cv2
import numpy as np
import copy
from image_registration import cross_correlation_shifts
from src.utils import draw_ball_curve, fill_lost_tracking
from src.FrameInfo import FrameInfo


def generate_overlay(
    video_frames: list[list[FrameInfo]],
    width: int,
    height: int,
    fps: int,
    outputPath: str,
    show_preview: bool = True,
) -> None:
    print("Saving overlay result to", outputPath)
    
    codecs_to_try = [
        ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),
        ("XVID", cv2.VideoWriter_fourcc(*"XVID")),
        ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
    ]
    
    out = None
    codec_name = None
    for name, codec in codecs_to_try:
        try:
            # Use original fps for output video (removed /2 which was halving the frame rate)
            out = cv2.VideoWriter(outputPath, codec, fps, (width, height))
            if out.isOpened():
                codec_name = name
                print(f"使用編解碼器：{codec_name}")
                break
        except Exception as e:
            print(f"編解碼器 {name} 失敗：{e}")
            if out:
                out.release()
            out = None
    
    if out is None or not out.isOpened():
        raise RuntimeError(f"無法建立輸出影片檔案：{outputPath}\n可能是編解碼器不支援或路徑無寫入權限。")

    frame_lists = sorted(video_frames, key=len, reverse=True)
    balls_in_curves = [[] for i in range(len(frame_lists))]
    shifts = {}

    is_single_video = len(frame_lists) == 1

    # Take the longest frames as background
    for idx, base_frame in enumerate(frame_lists[0]):
        # Overlay frames 
        if is_single_video:
            background_frame = base_frame.frame.copy()
        else:
            background_frame = base_frame.frame.copy()
            for list_idx, frameList in enumerate(frame_lists[1:]):
                if idx < len(frameList):
                    overlay_frame = frameList[idx]
                else:
                    overlay_frame = frameList[len(frameList) - 1]

                alpha = 1.0 / (list_idx + 2)
                beta = 1.0 - alpha
                corrected_frame = image_registration(
                    background_frame, overlay_frame, shifts, list_idx, width, height
                )
                background_frame = cv2.addWeighted(
                    corrected_frame, alpha, background_frame, beta, 0
                )

                # Prepare balls to draw
                if overlay_frame.ball_in_frame:
                    balls_in_curves[list_idx + 1].append(
                        [
                            overlay_frame.ball[0],
                            overlay_frame.ball[1],
                            overlay_frame.ball_color,
                        ]
                    )

        
        if base_frame.ball_in_frame:
            balls_in_curves[0].append(
                [base_frame.ball[0], base_frame.ball[1], base_frame.ball_color]
            )

        if not is_single_video:
            # Emphasize base frame
            base_frame_weight = 0.55
            background_frame = cv2.addWeighted(
                base_frame.frame,
                base_frame_weight,
                background_frame,
                1 - base_frame_weight,
                0,
            )

        # Draw transparent curve and non-transparent balls
        for trajectory in balls_in_curves:
            # Draw the last small tail, make the trajectory length and ball speed fit, so that the trajectory will not appear suddenly
            background_frame = draw_ball_curve(background_frame, trajectory, max_points=25)

        result_frame = cv2.cvtColor(background_frame, cv2.COLOR_RGB2BGR)

        if show_preview:
            try:
                cv2.imshow("result_frame", result_frame)
                if cv2.waitKey(60) & 0xFF == ord("q"):
                    break
            except Exception as e:
                # If the preview window fails (e.g. in a non-GUI environment), continue execution but do not show
                print(f"無法顯示預覽視窗（可忽略）：{e}")

        try:
            out.write(result_frame)
        except Exception as e:
            print(f"寫入影片 frame 時發生錯誤：{e}")
            break
    
    # Release resources
    if out:
        try:
            out.release()
        except Exception as e:
            print(f"警告：釋放影片寫入器時發生錯誤：{e}")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def image_registration(
    ref_image: np.ndarray,
    offset_image: FrameInfo,
    shifts: dict,
    list_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    # The shift is calculated once for each video and stored
    if list_idx not in shifts:
        xoff, yoff = cross_correlation_shifts(
            ref_image[:, :, 0], offset_image.frame[:, :, 0]
        )
        shifts[list_idx] = (xoff, yoff)
    else:
        xoff, yoff = shifts[list_idx]

    offset_image.ball = tuple(
        [offset_image.ball[0] - int(xoff), offset_image.ball[1] - int(yoff)]
    )
    matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
    corrected_image = cv2.warpAffine(offset_image.frame, matrix, (width, height))

    return corrected_image
