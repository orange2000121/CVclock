```
circle[0], circle[1] : 圓心x,y
circle[2] : 圓半徑
line[0], line[1] : 第一點x,y
line[2], line[3] : 第二點x,y
clock : (x, y, w, h)
```
## cascade classifier訓練步驟
1. opencv_annotation --annotations=pos2.txt --images=train2/positive/
> 框選訓練物件
2. opencv_createsamples -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
> 增加訓練樣本
3. opencv_traincascade -data ../cascade/ -vec pos.vec -bg neg.txt -w 24 -h 24 -numPos 50 -numNeg 50 numStages 10
> 每次使用50個正樣本，50個負樣本訓練10回。
> 
> w、h必須與createsamples的值一樣
## 資料結構
```
project
│
└───cascade(創建資料夾)
│   │   cascade.xml(生成檔案)
│
└───train  
    │
    └───positive(蒐集樣本)
    │   │   pos_train1.jpg
    │   │   pos_train2.jpg
    │   
    └───negative(蒐集樣本)
    │   │   neg_train1.jpg
    │   │   neg_train2.jpg
    │
    │   pos.txt(生成檔案)
    │   neg.txt(生成檔案)
    │   pos.vec(生成檔案)
```
### 參考資料
[openCV Cascade Classifier (YouTube)](https://www.youtube.com/watch?v=XrCAvs9AePM&t=55s&ab_channel=LearnCodeByGaming)