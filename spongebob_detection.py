import cv2 as cv
import numpy as np

cap = cv.VideoCapture(r'C:\Users\Midor\PycharmProjects\pythonProject\venv\videos\spanch_bob.mp4')
low_range = np.array([10, 100, 100], dtype=np.uint8)
upper_range = np.array([30, 205, 120], dtype=np.uint8)

# def nothing():
#     pass
#
# #find our low and upper values
#
# cv.namedWindow('result')
#
# cv.createTrackbar('minh', 'result', 0, 255, nothing)
# cv.createTrackbar('mins', 'result', 0, 255, nothing)
# cv.createTrackbar('minv', 'result', 0, 255, nothing)
#
# cv.createTrackbar('maxh', 'result', 0, 255, nothing)
# cv.createTrackbar('maxs', 'result', 0, 255, nothing)
# cv.createTrackbar('maxv', 'result', 0, 255, nothing)


while True:
    ret, frame = cap.read()
    #frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    frame = cv.GaussianBlur(frame, (7, 7), 0)

    # minh = cv.getTrackbarPos('minh', 'result')
    # mins = cv.getTrackbarPos('mins', 'result')
    # minv = cv.getTrackbarPos('minv', 'result')
    #
    # maxh = cv.getTrackbarPos('maxh', 'result')
    # maxs = cv.getTrackbarPos('maxs', 'result')
    # maxv = cv.getTrackbarPos('maxv', 'result')

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # mask = cv.inRange(frame, (minh, mins, minv), (maxh, maxs, maxv))
    mask = cv.inRange(hsv, low_range, upper_range)
    #cv.imshow('mask', mask)

    result = cv.bitwise_and(frame, frame, mask=mask)
    #cv.imshow('bitwise_and', result)


    result_to_bgr = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(result_to_bgr, cv.COLOR_BGR2GRAY)

    histogram_c = np.sum(gray[gray.shape[0] // 2:, :], axis=0)
    histogram_r = np.sum(gray[:, gray.shape[1] // 2:], axis=1)

    indx = np.argmax(histogram_c)
    indy = np.argmax(histogram_r)


    rectangle = frame.copy()
    # cv.circle(circle, (indx, indy), 10, (0, 0, 200), -1)
    cv.rectangle(rectangle, (indx-90, indy-90), (indx + 90, indy + 90), (0, 0, 200), thickness=3)
    cv.putText(rectangle, 'Spongebob', (indx, indy), cv.FONT_HERSHEY_PLAIN, 2, (20, 195, 239), 3)
    cv.imshow('circle', rectangle)


    if cv.waitKey(10) & 0xFF == ord('q'):
        print('END')
        break


cap.release()
cv.destroyAllWindows()
