import sys
import cv2 as cv


def face_detect(video_source, cascasdepath="haarcascade_frontalface_default.xml"):
    """
    """
    # load classifier model
    face_cascade = cv.CascadeClassifier(cascasdepath)
   
    cap = cv.VideoCapture(video_source)
    
    num_faces = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor = 1.2,
                                             minNeighbors = 5, minSize = (40,40))
        num_faces = len(faces)
        print(f'The number of faces found = {num_faces}')


        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+h, y+h), (0, 255, 0), 2)
            cv.imshow("Face detection demo", frame)
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return num_faces


if __name__ == "__main__":
    
    vide_source = 1
    face_detect(vide_source)