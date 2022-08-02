import cv2
import numpy as np
import math
import time
import statistics


def findCircle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    minDist = 100
    param1 = 40  # 500
    param2 = 50  # 200 #smaller value-> more false circles
    minRadius = 80
    maxRadius = 200  # 10

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    return circles


def findLines(img):
    dst = cv2.Canny(img, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    rho = 1
    theta = np.pi / 180
    threshold = 50
    lines = None
    min_line_length = 50
    max_line_gap = 10
    linesP = cv2.HoughLinesP(dst, rho, theta, threshold,
                             lines, min_line_length, max_line_gap)

    return img, linesP


def drawCircle(img, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    return img


def drawLine(frame, lines):
    if len(lines) > 0:
        for line in lines:
            cv2.line(frame, (line[0], line[1]), (line[2],
                     line[3]), (0, 0, 255), 3, cv2.LINE_AA)
    return frame


def showCircleAndLine(frame):
    circles = findCircle(frame)
    frame, linesP = findLines(frame)
    linesincircle = []
    # --------------------------------- 判斷線是否在圓圈內 -------------------------------- #
    if linesP is not None:
        for i in range(0, len(linesP)):
            line = linesP[i][0]
            if circles is not None:
                # line in circle
                for circle in circles[0, :]:
                    p1tocirclelength = math.sqrt(
                        (line[0] - circle[0])**2 + (line[1] - circle[1])**2)  # p1到圓心距離
                    p2tocirclelength = math.sqrt(
                        (line[2] - circle[0])**2 + (line[3] - circle[1])**2)  # p2到圓心距離
                    if (p1tocirclelength < circle[2]) and (p2tocirclelength < circle[2]+5):#判斷在園內
                        if(p1tocirclelength < 20 ):#判斷在圓心
                            linesincircle.append(line)
                            break
                        elif(p2tocirclelength < 20):
                            #change p1 and p2
                            p1temp =[line[0],line[1]]
                            line[0],line[1] = line[2],line[3]
                            line[2],line[3] = p1temp[0],p1temp[1]
                            linesincircle.append(line)
                            break
                
            
    for line in linesincircle:
        print(line)
    findHourMinSec(linesincircle,frame)
    frame = drawLine(frame, linesincircle)
    frame = drawCircle(frame, circles)
    # 顯示圖片
    cv2.imshow('frame', frame)


def findHourMinSec(lines,frame):
    lines_sort = []
    lines_len = []  # 每條線的長度
    if lines is not None:
        # ---------------------------------------------------------------------------- #
        #                                   找出時針分針秒針                            #
        # ---------------------------------------------------------------------------- #

        # -------------------------------- 將每條線從大到小排序 -------------------------------- #
        for line in lines:
            length = math.sqrt((line[0]-line[2])**2+(line[1]-line[3])**2)
            lines_len.append(length)
            if len(lines_sort) == 0:
                lines_sort.append((length, line))
                continue
            for lenidx in range(len(lines_sort)):
                if length > lines_sort[lenidx][0]:
                    lines_sort.insert(lenidx, (length, line))
                    break
                if lenidx == len(lines_sort)-1:
                    lines_sort.append((length, line))
        # ----------------------------------- 長度分群 ----------------------------------- #
        dividing_idx = [0,0]  # 分群點 [座標]
        difs=[0,0]  # 分群點 [差距]
        for i in range(len(lines_sort)-1):
            dif = lines_sort[i][0] - lines_sort[i+1][0]
            if dif > difs[1]:
                dividing_idx[1] = i
                difs[1] = dif
            elif dif > difs[0]:
                dividing_idx[0] = i
                difs[0] = dif
        print(dividing_idx)
        for line in lines_sort:
            print(line)
        # ---------------------------------------------------------------------------- #
        #                                     判斷時分秒                               #
        # ---------------------------------------------------------------------------- #
        sec_line = lines_sort[dividing_idx[0]]
        min_line = lines_sort[dividing_idx[1]]
        hour_line = lines_sort[dividing_idx[1]+1]
        cv2.putText(frame, "sec", ((sec_line[1][0]+sec_line[1][2])//2, (sec_line[1][1]+sec_line[1][3])//2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "min", ((min_line[1][0]+min_line[1][2])//2, (min_line[1][1]+min_line[1][3])//2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "hour", ((hour_line[1][0]+hour_line[1][2])//2, (hour_line[1][1]+hour_line[1][3])//2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv2.LINE_AA)

        sec_tan = -(sec_line[1][3] - sec_line[1][1]) / (sec_line[1][2] - sec_line[1][0])  # 轉換tan
        min_tan = -(min_line[1][3] - min_line[1][1]) / (min_line[1][2] - min_line[1][0])
        hour_tan = -(hour_line[1][3] - hour_line[1][1]) / (hour_line[1][2] - hour_line[1][0])

        # sec_angle = math.atan(sec_tan)   # 轉換成時鐘的角度
        # min_angle = math.atan(min_tan) 
        # hour_angle = math.atan(hour_tan)
        sec_angle = 2*math.pi - math.atan(sec_tan) + math.pi/2  # 轉換成時鐘的角度
        min_angle = 2*math.pi - math.atan(min_tan) + math.pi/2
        hour_angle = 2*math.pi - math.atan(hour_tan) + math.pi/2
        print("sec_angle:", math.degrees(sec_angle))
        print("min_angle:", math.degrees(min_angle))
        print("hour_angle:", math.degrees(hour_angle))
        # ---------------------------------- 調整為正確角度 --------------------------------- #
        if(sec_line[1][2]-sec_line[1][0] < 0 and sec_angle>0):
            sec_angle = sec_angle + math.pi
        elif(sec_line[1][2]-sec_line[1][0] > 0 and sec_angle<0):
            sec_angle = sec_angle - math.pi
        if(min_line[1][2]-min_line[1][0] < 0 and min_angle>0):
            min_angle = min_angle + math.pi
        elif(min_line[1][2]-min_line[1][0] > 0 and min_angle<0):
            min_angle = min_angle - math.pi
        if(hour_line[1][2]-hour_line[1][0] < 0 and hour_angle>0):
            hour_angle = hour_angle + math.pi
        elif(hour_line[1][2]-hour_line[1][0] > 0 and hour_angle<0):
            hour_angle = hour_angle - math.pi
        # ---------------------------------- 角度計算時間 ---------------------------------- #
        sec_time = int(sec_angle/(2*math.pi/60)) % 60
        min_time = int(min_angle/(2*math.pi/60)) % 60
        hour_time = int(hour_angle/(2*math.pi/12)) % 12
        print("時間: %d:%d:%d" % (hour_time, min_time, sec_time))


def cameraCap():
    # 選擇第二隻攝影機
    cap = cv2.VideoCapture(0)

    while(True):
        time.sleep(2)
        ret, frame = cap.read()  # 從攝影機擷取一張影像
        try:
            showCircleAndLine(frame)
        except:
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 若按下 q 鍵則離開迴圈
            break
    cap.release()  # 釋放攝影機
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗


def pictureCap():
    img = cv2.imread('149879.jpg')  # 讀取圖片
    img = cv2.resize(img, None, fx=0.3, fy=0.3,interpolation=cv2.INTER_AREA)  # 按比例縮小圖片0.3倍
    # img = cv2.Canny(img, 100, 200)  # 取得輪廓
    # try:
    #     showCircleAndLine(img)
    # except:
    #     cv2.imshow('frame', img)
    showCircleAndLine(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # pictureCap()
    cameraCap()
