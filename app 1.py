import os, cv2, pickle
import face_recognition
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
STATIC_DIR = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(STATIC_DIR, exist_ok=True)

KNN_ACCEPT_DIST = 0.6

# Load KNN + labels
with open(os.path.join(TRAINER_DIR, "knn_model.pkl"), "rb") as f:
    knn = pickle.load(f)
with open(os.path.join(TRAINER_DIR, "labels.pickle"), "rb") as f:
    label_map = pickle.load(f)

def build_id_to_name(label_map_obj, knn_obj):
    if all(isinstance(k, str) for k in label_map_obj.keys()):
        return {int(v): k for k, v in label_map_obj.items()}
    if all(isinstance(k, (int, np.integer)) for k in label_map_obj.keys()):
        return {int(k): v for k, v in label_map_obj.items()}
    classes = [int(c) for c in getattr(knn_obj, "classes_", [])]
    possible_names = [v for v in label_map_obj.values() if isinstance(v, str)]
    if len(possible_names) == len(classes):
        return {cid: nm for cid, nm in zip(classes, possible_names)}
    return {cid: f"ID_{cid}" for cid in classes}

id_to_name = build_id_to_name(label_map, knn)
all_students = sorted(set(id_to_name.values()))

# Global variables to handle step-by-step flow
current_stage = 'upload'  # stages: upload -> train -> attendance
last_uploaded_file = None
last_attendance = None
last_csv_file = None

@app.route("/", methods=["GET"])
def index():
    global current_stage
    return render_template(
        "index.html",
        stage=current_stage,
        processed_img=None if current_stage == 'upload' else last_uploaded_file,
        attendance=None if current_stage != 'attendance' else last_attendance,
        csv_file=None if current_stage != 'attendance' else last_csv_file,
        upload_message=None,
        train_message=None
    )

@app.route("/upload", methods=["POST"])
def upload():
    global current_stage, last_uploaded_file, last_attendance, last_csv_file

    file = request.files.get("group_image")
    if not file or file.filename == "":
        return render_template(
            "index.html",
            stage='upload',
            upload_message="No file selected"
        )

    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    filepath = os.path.join(STATIC_DIR, filename)
    file.save(filepath)

    last_uploaded_file = url_for("static", filename=f"outputs/{filename}")
    current_stage = 'train'  # move to next stage
    last_attendance = None

    # CSV file placeholder
    last_csv_file = None

    return render_template(
        "index.html",
        stage='train',
        processed_img=last_uploaded_file,
        upload_message=f"File '{file.filename}' uploaded successfully"
    )

@app.route("/train", methods=["POST"])
def train():
    global current_stage, last_uploaded_file, last_attendance, last_csv_file

    # Call your existing trainer function here
    # train()  # <-- keep your existing trainer code if any

    # Process attendance using the uploaded file
    filename = os.path.basename(last_uploaded_file)
    filepath = os.path.join(STATIC_DIR, filename)

    image = cv2.imread(filepath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model="hog")
    if not face_locations:
        small_rgb = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)
        small_locs = face_recognition.face_locations(small_rgb, model="cnn")
        face_locations = [(t*2, r*2, b*2, l*2) for (t,r,b,l) in small_locs]

    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    recognized = set()

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        try:
            pred_id = int(knn.predict([enc])[0])
            dists, idxs = knn.kneighbors([enc], n_neighbors=1)
            if dists[0][0] <= KNN_ACCEPT_DIST and pred_id in id_to_name:
                name = id_to_name[pred_id]
            else:
                name = "Unknown"
        except Exception:
            name = "Unknown"

        if name != "Unknown":
            recognized.add(name)

        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save labeled output
    out_path = os.path.join(STATIC_DIR, f"labeled_{filename}")
    cv2.imwrite(out_path, image)
    last_uploaded_file = url_for("static", filename=f"outputs/labeled_{filename}")

    # Attendance marking
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    csv_filename = os.path.join(STATIC_DIR, f"attendance_{date_str}.csv")  # SAVE IN STATIC

    rows, attendance = [], []
    for student in all_students:
        status = "Present" if student in recognized else "Absent"
        rows.append(f"{student},{date_str},{time_str},{status}")
        attendance.append((student, status))

    write_header = not os.path.exists(csv_filename)
    with open(csv_filename, "a", encoding="utf-8") as f:
        if write_header:
            f.write("Name,Date,Time,Status\n")
        for row in rows:
            f.write(row + "\n")

    last_attendance = attendance
    last_csv_file = url_for("static", filename=f"outputs/attendance_{date_str}.csv")
    current_stage = 'attendance'

    return render_template(
        "index.html",
        stage='attendance',
        processed_img=last_uploaded_file,
        attendance=last_attendance,
        csv_file=last_csv_file,
        train_message="Training completed successfully!"
    )

if __name__ == "__main__":
    app.run(debug=True)
