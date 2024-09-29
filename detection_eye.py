import cv2

# 画像読み込み
image_path = '読み込みたい画像'  
image = cv2.imread(image_path)

# 目の検出用分類器
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 目の検出
eyes = eye_cascade.detectMultiScale(gray)

for (x,y,w,h) in eyes:
    # 検知した顔を矩形で囲む
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    # 顔画像（グレースケール）
    roi_gray = gray[y:y+h, x:x+w]
    # 顔ｇ増（カラースケール）
    roi_color = image[y:y+h, x:x+w]
    # 顔の中から目を検知
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # 検知した目を矩形で囲む
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',image)

cv2.waitKey(0)
cv2.destroyAllWindows()
