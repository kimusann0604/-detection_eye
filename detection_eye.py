import cv2

image_path = '読み込みたい画像'  
image = cv2.imread(image_path)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#haarcascade_eye.xmlを読み込まないといけない

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray)

for (x,y,w,h) in eyes:#顔の検出をしてから目の検出をする
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',image)

cv2.waitKey(0)
cv2.destroyAllWindows()
