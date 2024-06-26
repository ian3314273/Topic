from ultralytics import YOLO
import cv2 as cv
import numpy as np
from scipy.spatial.distance import euclidean
source_path = "C:/Coding/Topic_One/240212104239.avi"
model = YOLO('yolov8n.pt')

def detect_people(frame):
    results = model(frame)
    count = 0
    centers = []
    for result in results:
        for box in result.boxes:
            # is person or not
            if box.cls.item() == 0:
                count += 1
                center_x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                center_y = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                centers.append((center_x.item(), center_y.item()))
    return results, count, centers
  
def distance_cal(centers):
    threshold = 100
    crowding = False
    len_centers = len(centers)
    for i in range(len_centers):
        for j in range(i+1, len_centers):
            if euclidean(centers[i], centers[j]) < threshold:
                crowding = True
                return True

def main():
    cap = cv.VideoCapture(source_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_interval = int(fps * 10)  # process every 10 seconds
    frame_count = 0
    counts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results, person_count, centers = detect_people(frame)
        if frame_count % frame_interval == 0:
            counts.append(person_count)
            # count the average person count in the last 10 seconds
            if len(counts) > 6:
                counts.pop(0)
            avg_person_count = np.mean(counts)
        for result in results:
            for box in result.boxes:
                if box.cls.item() == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv.putText(frame, f'Detected: {person_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame, f'Avg (10s): {int(avg_person_count)}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        if distance_cal(centers) == True and person_count >= avg_person_count:
            cv.putText(frame, 'Crowding', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                       2, cv.LINE_AA)
        else:
            cv.putText(frame, 'Not Crowding', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                       2, cv.LINE_AA)

        cv.imshow('Frame', frame)
        frame_count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
