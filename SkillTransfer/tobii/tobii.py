import cv2
import tobiiresearch as tr

def capture_camera_feed(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    return cap

def get_gaze_data(eye_tracker):
    gaze_data = None

    def gaze_callback(gaze_point_data):
        nonlocal gaze_data
        gaze_data = gaze_point_data

    eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_callback, as_dictionary=True)
    return gaze_data

def overlay_gaze_on_frame(frame, gaze_data):
    if gaze_data:
        gaze_point = gaze_data['left_gaze_point_on_display_area']
        x = int(gaze_point[0] * frame.shape[1])
        y = int(gaze_point[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    return frame

def main():
    camera_feed = capture_camera_feed()
    eye_tracker = tr.find_all_eyetrackers()[0]

    while True:
        ret, frame = camera_feed.read()
        if not ret:
            break

        gaze_data = get_gaze_data(eye_tracker)
        
        # A側のディスプレイに元の映像を表示
        cv2.imshow("A's Display", frame)
        
        # B側のディスプレイに視線情報を重畳した映像を表示
        frame_with_gaze = overlay_gaze_on_frame(frame.copy(), gaze_data)
        cv2.imshow("B's Display", frame_with_gaze)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_feed.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
