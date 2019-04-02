import numpy as np
import cv2

cap = cv2.VideoCapture('demo5.mp4')
i=0
width=cap.get(3)
height=cap.get(4)
fps=cap.get(5)
time=cap.get(6)
frames=cap.get(7)
print('width : ',width)
print('height : ',height)
print('fps : ',fps)
print('time : ',time)
print('frames : ',frames)
flag=True
is_paused=False
cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
delay=0
while flag:
    ret, frame = cap.read()
    i+=1
    print(i)
    cv2.imshow('SiamMask',frame)
    d=1000+delay
    if d<=0:
        d=fps
    keys=cv2.waitKey(int((d)/fps))
    if keys == ord('f'):
        delay-=500
    if keys == ord('n'):
        delay=0
    if keys == ord('q'):
        flag=False
    if i>=frames:
        flag=False
try:
    init_rect = cv2.selectROI('SiamMask', frame, False, False)
    print(init_rect)
except:
    pass
cap.release()
cv2.destroyAllWindows()