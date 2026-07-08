import cv2
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def create_dataset():
    roll_no = input("Enter Roll Number: ")
    name = input("Enter Name: ")

    # Create a folder to store dataset if it doesn't exist
    dataset_path = 'dataset/'
    user_path = os.path.join(dataset_path, roll_no + "_" + name)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't access webcam.")
            break

        # Convert frame to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{user_path}/img_{count}.jpg", face)
            count += 1

        cv2.imshow('Creating Dataset - Press Q to Exit', frame)

        if count >= 100 or cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Dataset created successfully for {name} (Roll No: {roll_no})")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_dataset()
