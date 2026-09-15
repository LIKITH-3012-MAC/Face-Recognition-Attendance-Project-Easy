import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_DIR = "Training_images"
ATTENDANCE_FILE = "Attendance.csv"

# Lower = stricter recognition
# Start with 0.50 and tune using your camera/data.
TOLERANCE = 0.50

# Resize webcam frame for faster recognition
FRAME_SCALE = 0.25


# ============================================================
# CREATE REQUIRED FILES/FOLDERS
# ============================================================

if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])


# ============================================================
# LOAD REGISTERED STUDENTS
# ============================================================

def load_training_images():

    images = []
    student_names = []

    files = os.listdir(TRAINING_DIR)

    for file in files:

        # Accept only image files
        if not file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(TRAINING_DIR, file)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read: {file}")
            continue

        images.append(image)

        # Example:
        # Likith.jpg -> Likith
        name = os.path.splitext(file)[0]

        student_names.append(name)

    return images, student_names


# ============================================================
# GENERATE FACE ENCODINGS
# ============================================================

def generate_encodings(images, names):

    known_encodings = []
    known_names = []

    for image, name in zip(images, names):

        # OpenCV uses BGR
        # face_recognition expects RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_image
        )

        if len(face_locations) == 0:

            print(
                f"[WARNING] No face found in {name}"
            )

            continue

        if len(face_locations) > 1:

            print(
                f"[WARNING] Multiple faces found in {name}. "
                f"Use an image containing only {name}."
            )

            continue

        # Generate 128-dimensional face encoding
        encodings = face_recognition.face_encodings(
            rgb_image,
            face_locations
        )

        if len(encodings) == 0:
            continue

        known_encodings.append(encodings[0])
        known_names.append(name)

        print(
            f"[OK] Face encoding generated for {name}"
        )

    return known_encodings, known_names


# ============================================================
# ATTENDANCE
# ============================================================

def mark_attendance(name):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Read existing attendance
    with open(ATTENDANCE_FILE, "r", newline="") as f:

        reader = csv.reader(f)
        rows = list(reader)

    # Check whether this student already has attendance today
    for row in rows[1:]:

        if len(row) >= 2:

            existing_name = row[0]
            existing_date = row[1]

            if (
                existing_name.lower() == name.lower()
                and existing_date == today
            ):
                return

    # Add attendance
    with open(
        ATTENDANCE_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [name, today, current_time]
        )

    print(
        f"[ATTENDANCE] {name} marked present "
        f"at {current_time}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n====================================")
print("     SMART ATTENDANCE SYSTEM")
print("====================================\n")

images, names = load_training_images()

print("Registered students:")

for name in names:
    print(" -", name)

print(
    f"\nTotal registered students: {len(names)}\n"
)


# ============================================================
# GENERATE KNOWN FACE ENCODINGS
# ============================================================

known_encodings, known_names = generate_encodings(
    images,
    names
)

print("\nEncoding Complete!")

print(
    f"Valid face encodings: {len(known_encodings)}"
)

if len(known_encodings) == 0:

    print(
        "\nERROR: No valid face encodings found."
    )

    print(
        "Put student images inside:"
    )

    print(
        f"    {TRAINING_DIR}/"
    )

    exit()


# ============================================================
# START CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Cannot open webcam.")
    exit()


print("\nCamera started.")
print("Press Q to quit.\n")


# ============================================================
# LIVE RECOGNITION
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Failed to read webcam.")
        break


    # --------------------------------------------------------
    # Resize frame
    # --------------------------------------------------------

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=FRAME_SCALE,
        fy=FRAME_SCALE
    )


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    face_locations = face_recognition.face_locations(
        rgb_small_frame
    )


    # --------------------------------------------------------
    # Generate encodings for detected faces
    # --------------------------------------------------------

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )


    # ========================================================
    # PROCESS EACH FACE
    # ========================================================

    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        # ----------------------------------------------------
        # Calculate distance from every registered student
        # ----------------------------------------------------

        face_distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )


        # ----------------------------------------------------
        # Find closest registered student
        # ----------------------------------------------------

        best_match_index = np.argmin(
            face_distances
        )

        best_distance = face_distances[
            best_match_index
        ]


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Don't blindly accept nearest face.
        #
        # Check the distance against tolerance.
        # ----------------------------------------------------

        if best_distance <= TOLERANCE:

            name = known_names[
                best_match_index
            ]

            confidence = (
                (1 - best_distance) * 100
            )

            display_name = name.upper()

            mark_attendance(name)

            box_color = (0, 255, 0)

        else:

            name = "UNKNOWN"

            confidence = 0

            display_name = "UNKNOWN"

            box_color = (0, 0, 255)


        # ----------------------------------------------------
        # Convert coordinates back to original frame size
        # ----------------------------------------------------

        top, right, bottom, left = face_location

        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)


        # ----------------------------------------------------
        # Draw face rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            box_color,
            2
        )


        # ----------------------------------------------------
        # Draw label background
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (left, bottom - 40),
            (right, bottom),
            box_color,
            cv2.FILLED
        )


        # ----------------------------------------------------
        # Display name
        # ----------------------------------------------------

        cv2.putText(
            frame,
            display_name,
            (left + 6, bottom - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Display distance/confidence
        # ----------------------------------------------------

        if name != "UNKNOWN":

            distance_text = (
                f"Distance: {best_distance:.2f}"
            )

            cv2.putText(
                frame,
                distance_text,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Smart Attendance",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\nCamera stopped.")
print("Smart Attendance System closed.")
