import cv2, time
from datetime import datetime

first_img = None
status_list = [None, None]
times = []
video = cv2.VideoCapture(0)

while True:
    check, img = video.read()
    status = 0
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

    if first_img is None:
        first_img = gray_img
        continue

    delta_img = cv2.absdiff(first_img, gray_img)
    thresh_img = cv2.threshold(delta_img, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_img = cv2.dilate(thresh_img, None, iterations=2)

    image, contour, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contours in contour:
        if cv2.contourArea(contours) < 10000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contours)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Gray Frame", gray_img)
    cv2.imshow("Delta Frame", delta_img)
    cv2.imshow("Threshold Frame",thresh_img)
    cv2.imshow("Image Frame", img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # 'q' key
        if status == 1:
            times.append(datetime.now())
        break

print(times)
print(status_list)
video.release()
cv2.destroyAllWindows()