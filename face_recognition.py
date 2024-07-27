import cv2, numpy, os

size = 4
harr_file = "haarcascade_frontalface_default.xml"
datasets = 'datasets'

(images, labels, name, id) = ([], [], {}, 0)
for (subdir, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        name[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        #  print(subjectpath)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id;
            images.append(cv2.imread(path, 0))
            # print(images)
            labels.append(int(label))
        id += 1;

(width, height) = (130, 100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
# print(images,labels)
user=int(input("Enter 1 for training and 0 for not"))
if user==1:
    print("training")
    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(images, labels)
    print("Training complete")
    model.save('your_model.xml')

model = cv2.face.LBPHFaceRecognizer_create()
model.read('your_model.xml')
face_cascade = cv2.CascadeClassifier(harr_file)
webcam = cv2.VideoCapture(0)
cnt = 0
while True:
    _, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if prediction[1] > 60:
            cv2.putText(im, '%s- %.0f' % (name[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0))
            print(name[prediction[0]])
            cnt = 0

        else:
            cnt += 1
            cv2.putText(im, 'unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if (cnt > 100):
                print("unknown image")
                cv2.imwrite("input.jpg", im)
                cnt = 0;

        cv2.imshow("OpenCv", im)
    if cv2.waitKey(10) == 27:  # Exit on 'ESC' key
        break

webcam.release()
cv2.destroyAllWindows()

# Remainder of file ignored
# Collecting opencv-contrib-python==4.1.2.30