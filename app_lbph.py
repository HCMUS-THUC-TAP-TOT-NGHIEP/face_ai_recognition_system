
import os.path
import time
from PIL import Image
from threading import Thread
import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('datasets'):
    os.makedirs('datasets')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('datasets'))

# -------------- image labesl ------------------------

def getImagesAndLabels(path):
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    for folder in os.listdir(path):
        # get the path of all the files in the folder
        folder_path = os.path.join(path,folder)
        imagePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        # now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(Id)
    return faces, Ids

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

#### Add Attendance of a specific user
def add_attendance(id):
    for name in os.listdir('datasets'):
        if name.split('_')[1] == id:
            username = name.split('_')[0]
            userid = name.split('_')[1]
            current_time = datetime.now().strftime("%H:%M:%S")

            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            if int(userid) not in list(df['Roll']):
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')

### A function which trains the model on all the faces available in faces folder
def train_model():
    # ----------- train images function ---------------
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "static/haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("datasets")
    Thread(target=recognizer.train(faces, np.array(Id))).start()
    # Below line is optional for a visual counter effect
    Thread(target=counter_img("datasets")).start()
    recognizer.save("static/Trainner.yml")
    print("All Images")

# Optional, adds a counter for images trained (You can remove it)
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1

def get_name_from_ID(id):
    for folder in os.listdir('datasets'):
        name, str_id = folder.split('_')
        if int(str_id) == id:
            return name
#-------------------------start recognition ---------------------------------
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
            add_attendance(str(id))
            s = " id: " + str(id)
            name = get_name_from_ID(id) + s
            if confidence > 70:
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)


    def recognize(img, clf, faceCascade):
        draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), clf)

    faceCascade = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("static/Trainner.yml")

    wCam, hCam = 500, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        recognize(img, clf, faceCascade)
        cv2.imshow('Attendance',img)
        key = cv2.waitKey(1)
        if key == 27:
            break

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)

#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    if 'Trainner.yml' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.')

    face_recognition()
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)


@app.route('/add',methods=['GET','POST'])
def add():
    name = request.form['newusername']
    Id = request.form['newuserid']
    userimagefolder = 'datasets/' + name + '_' + str(Id)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cam = cv2.VideoCapture(0)
    harcascadePath = 'static/haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    maxSample = 50

    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder TrainingImage
            cv2.imwrite(userimagefolder + '/' + name + "." + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            # display the frame
            cv2.imshow('frame', img)
        # wait for 100 miliseconds
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 100
        elif sampleNum >= maxSample:
            break
    cam.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)

