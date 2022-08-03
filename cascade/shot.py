import cv2
import time

# get camera
cap = cv2.VideoCapture(0)
# shot 10 per second
start_time = time.time()
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    # store image in positive folder
    cv2.imwrite('train2/positive/' + str(time.time()) + '.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1/10)
    if time.time() - start_time > 30:
        break