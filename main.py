import cv2

from mosse import MOSSETracker
from utils import VideoBatchReader


record_result = False

if __name__ == "__main__":
    video = VideoBatchReader(0, False)
    tracker = MOSSETracker(sigma_amplitude=0.025, transforms_number=8, learning_rate=0.2,
                           psr_threshold=5.7, debug_mode=False)
    if record_result:
        out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
    for frame in video.frames():
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if tracker.is_inited():
            roi, ok = tracker.track(frame)
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 3)
            cv2.imshow("frame", frame)
            if record_result:
                out.write(frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            tracked_object_roi = cv2.selectROI('frame', frame, True, False)
            if tracked_object_roi[2] == 0 or tracked_object_roi[3] == 0:
                print("No object is selected, shutting down...")
                break
            tracker.initialize(frame, tracked_object_roi)
    if record_result:
        out.release()
    print("DONE")
