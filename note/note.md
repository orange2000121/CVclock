```
circle[0], circle[1] : 圓心x,y
circle[2] : 圓半徑
line[0], line[1] : 第一點x,y
line[2], line[3] : 第二點x,y
clock : (x, y, w, h)
```
## 步驟
1. opencv_annotation --annotations=pos2.txt --images train2/positive/
2. opencv_createsamples -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
3. opencv_traincascade -data ../cascade/ -vec pos.vec -bg neg.txt -w 24 -h 24 -numPos 50 -numNeg 50 numStages 10
### 參考資料
[openCV Cascade Classifier](https://www.youtube.com/watch?v=XrCAvs9AePM&t=55s&ab_channel=LearnCodeByGaming)