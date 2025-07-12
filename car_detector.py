import cv2

cascade_src = 'cars.xml'
video_src   = 'car.mp4'  # غيّره حسب اسم الفيديو عندك

car_cascade = cv2.CascadeClassifier(cascade_src)
cap = cv2.VideoCapture(video_src)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Car Detection', frame)
    if cv2.waitKey(33) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()