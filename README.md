<div align="center">

<!-- ========================= HERO ========================= -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:2563eb,100:06b6d4&height=220&section=header&text=Face%20Recognition%20Attendance&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Real-Time%20AI-Powered%20Student%20Attendance%20System&descAlignY=58&descSize=18" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=06B6D4&center=true&vCenter=true&width=850&lines=AI-Powered+Attendance+System;Real-Time+Face+Detection+%2B+Recognition;128-Dimensional+Face+Encodings;OpenCV+%2B+face_recognition+%2B+dlib;Automatic+Date+%26+Time+Attendance;Built+for+Learning%2C+Experimentation+%26+Engineering" alt="Typing SVG" />

<br/><br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/dlib-Face%20Recognition-111111?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AI-Face%20Recognition-06B6D4?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Encoding-128D-8B5CF6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Recognition-Euclidean%20Distance-F59E0B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Attendance-CSV-22C55E?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Learning%20Project-EC4899?style=for-the-badge"/>
</p>

<br/>

**A real-time AI-based attendance system that detects faces, generates 128-dimensional face encodings, compares them against registered students, and automatically records attendance with date and time.**

<br/>

<a href="#-overview">Overview</a> • <a href="#-architecture">Architecture</a> • <a href="#-installation">Installation</a> • <a href="#-usage">Usage</a> • <a href="#-how-it-works">How It Works</a> • <a href="#-troubleshooting">Troubleshooting</a>

</div>

---

# 🧠 Overview

**Face Recognition Attendance** is a real-time computer-vision project designed to automate student attendance using a webcam.

Instead of manually entering attendance, the system:

```text
📷 Webcam
   ↓
👤 Face Detection
   ↓
🧬 Face Encoding
   ↓
🔢 128-Dimensional Vector
   ↓
📐 Distance Comparison
   ↓
🎯 Best Match
   ↓
🔐 Tolerance Check
   ↓
✅ Recognized / ❌ Unknown
   ↓
📝 Attendance Record
```

The project uses a **pretrained face-recognition model** rather than training a neural network from scratch.

---

# ✨ Features

| Feature                  | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| 👤 Face Detection        | Detects faces inside webcam frames                            |
| 🧬 Face Encoding         | Converts faces into 128-dimensional numerical representations |
| 🎯 Face Recognition      | Compares live faces against registered students               |
| 📐 Distance Matching     | Uses face distance to find the closest known face             |
| 🔐 Tolerance             | Controls how strict recognition should be                     |
| ⚡ Real-Time Processing   | Processes webcam frames continuously                          |
| 📅 Date Tracking         | Records attendance date                                       |
| ⏰ Time Tracking          | Records attendance time                                       |
| 📝 CSV Storage           | Stores attendance records in `Attendance.csv`                 |
| 🖼️ Multi-Face Detection | Can process multiple detected faces in a frame                |
| 🟢 Visual Feedback       | Displays recognized faces on the camera                       |
| 🔴 Unknown Detection     | Marks faces that don't satisfy the threshold as unknown       |

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │    Student Image     │
                         │     Likith.jpg       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Face Detection     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Face Encoding      │
                         │      128D Vector     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Known Encodings    │
                         │       RAM            │
                         └──────────┬───────────┘
                                    │
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    │                               │
                    ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │   Webcam Frame   │            │ Known Encodings  │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   ▼                               │
          ┌──────────────────┐                     │
          │ Resize to 25%    │                     │
          └────────┬─────────┘                     │
                   │                               │
                   ▼                               │
          ┌──────────────────┐                     │
          │    BGR → RGB     │                     │
          └────────┬─────────┘                     │
                   │                               │
                   ▼                               │
          ┌──────────────────┐                     │
          │  Face Detection  │                     │
          └────────┬─────────┘                     │
                   │                               │
                   ▼                               │
          ┌──────────────────┐                     │
          │   Live 128D      │                     │
          │    Encoding      │                     │
          └────────┬─────────┘                     │
                   │                               │
                   └──────────────┬────────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Euclidean Distance  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │    np.argmin()     │
                       │    Best Candidate   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Tolerance Check    │
                       │   ≤ 0.50 ?          │
                       └─────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              ┌──────────┐             ┌──────────┐
              │   MATCH  │             │ UNKNOWN  │
              └────┬─────┘             └──────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Mark Attendance  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Attendance.csv   │
          └──────────────────┘
```

---

# 🔬 How It Works

## 1. Student Registration

A student's reference image is placed inside:

```text
Training_images/
```

Example:

```text
Training_images/
├── Likith.jpg
├── Rahul.jpg
└── Suresh.jpg
```

The filename is used as the student's identity.

For example:

```text
Likith.jpg
```

becomes:

```text
Likith
```

---

## 2. Image Loading

OpenCV loads the image into memory.

```python
image = cv2.imread(image_path)
```

The image is represented as numerical pixel data.

---

## 3. BGR → RGB

OpenCV normally works with:

```text
BGR
```

while the `face_recognition` pipeline expects:

```text
RGB
```

Therefore:

```python
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

---

# 👤 Face Detection

The system first answers:

> **WHERE is the face?**

Using:

```python
face_locations = face_recognition.face_locations(rgb_image)
```

A detected face is represented using coordinates:

```text
(top, right, bottom, left)
```

Detection and recognition are two different operations.

```text
Face Detection
      ↓
WHERE is the face?

Face Recognition
      ↓
WHO is the face?
```

---

# 🧬 Face Encoding

After detecting a face, the system generates a numerical representation:

```python
face_encodings = face_recognition.face_encodings(
    rgb_image,
    face_locations
)
```

The output is a **128-dimensional face encoding**.

Conceptually:

```text
Face
 ↓
Pretrained Neural Network
 ↓
[0.12, -0.08, 0.44, 0.21, ...]
 ↓
128 numerical values
```

This vector is also commonly described as a **face embedding**.

---

# 🧠 What Is the 128D Representation?

The 128D vector is **not the model name**.

The underlying pretrained model is:

```text
dlib_face_recognition_resnet_model_v1
```

The model is based on a ResNet-style neural network and is designed using metric learning.

Conceptually:

```text
Input Face
     ↓
ResNet-based Network
     ↓
Feature Representation
     ↓
128D Vector
```

The goal is to create a numerical space where:

```text
Same person
     ↓
Vectors are closer

Different people
     ↓
Vectors are farther apart
```

---

# 📐 Face Distance

To recognize a face, the live encoding is compared against all registered encodings.

```python
face_distances = face_recognition.face_distance(
    known_encodings,
    face_encoding
)
```

Conceptually:

```text
Live Face
   │
   ├── compare → Likith
   ├── compare → Rahul
   └── compare → Suresh
```

The system calculates a distance for each comparison.

Example:

```text
Likith → 0.31
Rahul  → 0.72
Suresh → 0.84
```

Smaller distance means greater similarity.

---

# 🎯 Best Match

The project uses NumPy:

```python
best_match_index = np.argmin(face_distances)
```

`np.argmin()` returns the index of the smallest distance.

Example:

```text
[0.31, 0.72, 0.84]
   ↑
 smallest
```

Therefore:

```text
best_match_index = 0
```

If:

```python
known_names[0] == "Likith"
```

the closest candidate is:

```text
Likith
```

---

# 🔐 Tolerance

The project uses:

```python
TOLERANCE = 0.50
```

The decision is approximately:

```text
distance ≤ 0.50
        ↓
      MATCH

distance > 0.50
        ↓
     UNKNOWN
```

### Important

`0.50` does **not** mean:

```text
50% confidence
```

It is a **distance threshold**.

---

# ⚖️ Tolerance Trade-Off

A lower tolerance makes the system stricter.

```text
Lower tolerance
      ↓
Fewer false matches
      ↓
May reject genuine faces
```

A higher tolerance makes the system more permissive.

```text
Higher tolerance
      ↓
May accept more genuine variations
      ↓
May increase false matches
```

Therefore, threshold selection is a trade-off between:

```text
False Positive
vs
False Negative
```

---

# ⚡ Real-Time Processing

The webcam is opened using:

```python
camera = cv2.VideoCapture(0)
```

The application continuously captures frames:

```python
while True:
    success, frame = camera.read()
```

The loop is essentially:

```text
Capture
   ↓
Resize
   ↓
Convert
   ↓
Detect
   ↓
Encode
   ↓
Compare
   ↓
Display
   ↓
Repeat
```

---

# 🚀 Why Resize the Frame?

The project uses:

```python
FRAME_SCALE = 0.25
```

The frame is reduced to 25% of its original width and height before recognition.

Example:

```text
1920 × 1080
     ↓
  25% scale
     ↓
480 × 270
```

This reduces computation and improves real-time performance.

Trade-off:

```text
Smaller frame
    ↓
Less computation
    ↓
Faster processing

BUT

Smaller frame
    ↓
Less spatial detail
```

---

# 📝 Attendance Recording

After successful recognition, the system calls:

```python
mark_attendance(name)
```

Attendance information includes:

```text
Name
Date
Time
```

and is stored in:

```text
Attendance.csv
```

Example:

```csv
Name,Date,Time
Likith,2026-09-15,10:21:35
Rahul,2026-09-15,10:22:11
```

The attendance mechanism is designed to prevent repeatedly recording the same student's attendance within the intended attendance logic.

---

# 🛠️ Technology Stack

<div align="center">

| Technology          | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| 🐍 Python           | Core programming language                           |
| 👁️ OpenCV          | Image/video processing and webcam                   |
| 🧠 face_recognition | Face detection, encoding and distance comparison    |
| 🔬 dlib             | Underlying face-recognition neural network          |
| 🔢 NumPy            | Numerical operations and minimum-distance selection |
| 📄 CSV              | Attendance storage                                  |
| 🕒 datetime         | Date/time generation                                |
| 📁 os               | File and directory operations                       |

</div>

---

# 📁 Project Structure

```text
Face-Recognition-Attendance-Projects/
│
│
│
├── 📁 Training_images/
│   └── Student reference images
│
├── 📄 main.py
│   └── Main face-recognition attendance application
│
│
├── 📄 Attendance.csv
│   └── Attendance records
│
├── 📄 requirements.txt
│   └── Python dependencies
│
├── 📄 README.md
│   └── Project documentation
│
└── 📄 .gitignore
    └── Files excluded from Git
```

> **Privacy note:** `Training_images/`, `Attendance.csv`, generated vector databases, `.env`, and system files should not be committed when they contain personal or sensitive information.

---

# 💻 Requirements

Recommended:

```text
Python 3.x
Webcam
Git
pip
Virtual Environment
```

The project dependencies are listed in:

```text
requirements.txt
```

---

# 🧰 Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/LIKITH-3012-MAC/Face-Recognition-Attendance-Project-Easy.git
```

Move into the project:

```bash
cd Face-Recognition-Attendance-Project-Easy
```

---

# 🐍 2️⃣ Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

You should see something similar to:

```text
(venv) user@machine project %
```

---

### Windows

```powershell
python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

---

# 📦 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
```

Then:

```bash
pip install -r requirements.txt
```

If you are installing the packages manually:

```bash
pip install opencv-python
pip install face-recognition
pip install numpy
```

---

# 🍎 macOS Setup

On macOS, some face-recognition dependencies may require native build tools.

If necessary:

```bash
xcode-select --install
```

For Homebrew users:

```bash
brew update
```

Then install common build dependencies if your environment requires them:

```bash
brew install cmake
```

Verify:

```bash
cmake --version
```

---

# 📷 Camera Permission

macOS may require camera permission.

Go to:

```text
System Settings
    ↓
Privacy & Security
    ↓
Camera
    ↓
Enable access for your terminal / IDE
```

Then restart the terminal or IDE if necessary.

---

# 👤 Register a Student

Create:

```text
Training_images/
```

Then add a student's reference image.

Example:

```text
Training_images/
└── Likith.jpg
```

The filename becomes the identity:

```text
Likith.jpg
   ↓
Likith
```

### Recommended registration image

Use an image with:

* one clearly visible face
* reasonable lighting
* minimal obstruction
* front-facing or reasonably aligned face
* good image quality

Avoid registering an image containing multiple people.

---

# ▶️ Run the Project

Activate your environment:

```bash
source venv/bin/activate
```

Then:

```bash
python main.py
```

or on some systems:

```bash
python3 main.py
```

The webcam should open.

---

# 🎮 Runtime Controls

When the camera window is open:

```text
Q
↓
Quit application
```

The application releases the camera and closes OpenCV windows when terminated.

---

# 🔄 Complete Runtime Pipeline

```text
                 START
                   │
                   ▼
          Load Training Images
                   │
                   ▼
           Detect Registered Face
                   │
                   ▼
            Generate 128D Encoding
                   │
                   ▼
          Store Encoding in RAM
                   │
                   ▼
             Open Webcam
                   │
                   ▼
            Capture Frame
                   │
                   ▼
            Resize to 25%
                   │
                   ▼
              BGR → RGB
                   │
                   ▼
             Detect Faces
                   │
                   ▼
       Generate Live Face Encoding
                   │
                   ▼
         Calculate Face Distances
                   │
                   ▼
              np.argmin()
                   │
                   ▼
          Find Closest Candidate
                   │
                   ▼
          Compare With Tolerance
                   │
             ┌─────┴─────┐
             ▼           ▼
          MATCH       UNKNOWN
             │
             ▼
      Mark Attendance
             │
             ▼
      Attendance.csv
             │
             ▼
          Next Frame
             │
             └───────────────↺
```

---

# 🧠 Detection vs Encoding vs Recognition

This distinction is fundamental.

### Face Detection

```text
Question:
WHERE is the face?
```

Output:

```text
(top, right, bottom, left)
```

---

### Face Encoding

```text
Question:
HOW can the face be represented numerically?
```

Output:

```text
128-dimensional vector
```

---

### Face Recognition

```text
Question:
WHO is the person?
```

Process:

```text
Live 128D Vector
       ↓
Compare with known vectors
       ↓
Calculate distances
       ↓
Find minimum distance
       ↓
Apply tolerance
       ↓
Identity
```

---

# 🧬 Pretrained Model

This project does **not** train the face-recognition neural network from scratch every time the application runs.

It uses a pretrained model:

```text
dlib_face_recognition_resnet_model_v1
```

Conceptually:

```text
face_recognition
       ↓
      dlib
       ↓
dlib_face_recognition_resnet_model_v1
       ↓
ResNet-based network
       ↓
128D face representation
```

The model is designed around metric learning.

The important idea is:

```text
Same person
→ closer vectors

Different people
→ farther vectors
```

---

# 💾 Where Are Encodings Stored?

In the current application architecture, the generated known encodings are held in Python memory:

```python
known_encodings
```

Conceptually:

```text
known_encodings
│
├── Likith → 128D
├── Rahul  → 128D
└── Suresh → 128D
```

They are **not automatically persisted as a permanent model/database by the main recognition flow**.

Therefore:

```text
Program starts
      ↓
Images loaded
      ↓
Encodings generated
      ↓
Stored in RAM
```

When the program exits:

```text
RAM encodings → released
```

On the next execution, the encodings are generated again from the registered images.

---

# 🔐 Privacy & Security

Face data is biometric information and should be handled carefully.

Do **NOT** commit:

```text
Training_images/
Attendance.csv
face_vector_db/
.env
API keys
private credentials
personal datasets
```

A suitable `.gitignore` includes:

```gitignore
Training_images/
face_vector_db/
Attendance.csv
.DS_Store
__pycache__/
*.pyc
.env
```

---

# 🚨 Important Git Safety

Before pushing:

```bash
git status
```

Check that private images and attendance records are not staged.

Then:

```bash
git add .
```

Check again:

```bash
git status
```

Commit:

```bash
git commit -m "Update project documentation"
```

Push:

```bash
git push origin main
```

---

# 🔧 Useful Git Commands

Initialize repository:

```bash
git init
```

Check status:

```bash
git status
```

See remotes:

```bash
git remote -v
```

Add remote:

```bash
git remote add origin <repository-url>
```

Change remote:

```bash
git remote set-url origin <repository-url>
```

Create/switch main branch:

```bash
git branch -M main
```

Stage everything:

```bash
git add .
```

Commit:

```bash
git commit -m "your message"
```

Push:

```bash
git push -u origin main
```

Pull:

```bash
git pull origin main
```

View commits:

```bash
git log --oneline
```

---

# 🧹 Common Cleanup Commands

Remove a tracked file but keep it locally:

```bash
git rm --cached filename
```

Remove tracked directory but keep locally:

```bash
git rm -r --cached directory_name
```

Example:

```bash
git rm -r --cached Training_images
```

Then:

```bash
git add .
git commit -m "Update gitignore"
```

---

# 🐛 Troubleshooting

## ❌ `zsh: command not found`

Example:

```text
zsh: command not found: l
```

This simply means `l` is not configured as a shell alias/command.

Use:

```bash
ls
```

Similarly, don't type explanatory comments such as:

```bash
# chudu
```

unless you actually intend to enter a shell comment.

---

## ❌ Camera Doesn't Open

Check:

```python
cv2.VideoCapture(0)
```

Try:

```python
cv2.VideoCapture(1)
```

if another camera is being used as the default device.

Also check OS camera permissions.

---

## ❌ No Face Detected

Possible causes:

```text
Poor lighting
Face too small
Face partially blocked
Low-quality image
Incorrect image format
Multiple faces in registration image
```

Try a clearer reference image.

---

## ❌ Recognition Is Too Strict

If genuine faces are being rejected:

```text
distance > tolerance
```

The threshold may be too strict for the environment.

Test carefully before changing:

```python
TOLERANCE = 0.50
```

Do not blindly increase it.

---

## ❌ Too Many False Matches

If unknown people are being recognized incorrectly, the threshold may be too permissive.

Consider:

```text
Better reference images
Better lighting
More consistent face capture
Stricter threshold
```

---

## ❌ `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'face_recognition'
```

Activate your virtual environment:

```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

Verify:

```bash
pip list
```

---

# 📊 Performance Considerations

The project uses:

```python
FRAME_SCALE = 0.25
```

to reduce the computational cost of processing webcam frames.

The basic trade-off is:

```text
Higher resolution
    ↓
More detail
    ↓
More computation

Lower resolution
    ↓
Less computation
    ↓
Faster processing
```

Real-time face recognition therefore involves balancing:

```text
Speed
Accuracy
Resolution
Hardware
Lighting
Threshold
```

---

# ⚠️ Limitations

This project is an educational and experimental implementation.

Real-world performance can be affected by:

* lighting conditions
* camera quality
* face angle
* facial occlusion
* image quality
* distance from camera
* multiple faces
* threshold selection
* hardware performance

Face recognition should therefore not be treated as mathematically perfect identity verification.

---

# 🚀 Future Improvements

Possible next-generation improvements include:

```text
☁️ Cloud deployment
🗄️ PostgreSQL / Supabase database
🔐 Authentication & authorization
📊 Attendance dashboard
📈 Attendance analytics
📧 Email notifications
📍 Location-aware attendance
📱 Mobile/PWA interface
🧠 Better embedding storage
⚡ GPU acceleration
🎥 Liveness detection
🛡️ Anti-spoofing
🔎 Advanced search
📤 PDF/Excel export
👨‍🎓 Student management
👨‍🏫 Faculty dashboard
📅 Attendance reports
🔔 Real-time notifications
```

---

# 🧪 Engineering Concepts Demonstrated

This project provides practical exposure to:

```text
Python
│
├── File Handling
├── Functions
├── Lists
├── Loops
├── Exception Handling
└── Modules

Computer Vision
│
├── Image Processing
├── Color Spaces
├── Video Capture
├── Face Detection
└── Real-Time Processing

Machine Learning
│
├── Pretrained Models
├── Face Embeddings
├── Metric Learning
├── Vector Representations
└── Distance-Based Matching

Mathematics
│
├── Vectors
├── Euclidean Distance
├── Minimum Search
└── Thresholding

Software Engineering
│
├── Git
├── GitHub
├── Virtual Environments
├── Dependency Management
└── Project Documentation
```

---

# 🎓 Viva / Interview Explanation

### "Explain your project."

> **This project is a real-time AI-based face recognition attendance system. It uses OpenCV for webcam and image processing, while the face_recognition library uses a pretrained dlib face-recognition model to detect faces and generate 128-dimensional face encodings. During registration, the reference encodings are generated and kept in memory. During live recognition, the webcam frame is processed, a 128D encoding is generated for each detected face, and its Euclidean distance is calculated against the registered encodings. The closest candidate is selected using NumPy's `argmin`, and the match is accepted only when the distance satisfies the configured tolerance. Once recognized, the student's attendance is recorded with date and time in a CSV file.**

---

# ❓ Common Viva Questions

### Q1. What is face detection?

Finding the location of a face in an image.

### Q2. What is face recognition?

Determining which registered identity a detected face most closely matches.

### Q3. What is a face encoding?

A numerical representation of a face.

### Q4. Why 128 dimensions?

The pretrained dlib face-recognition network used by the `face_recognition` ecosystem produces a 128-dimensional face descriptor.

### Q5. Is 128D the model?

No.

```text
Model = dlib_face_recognition_resnet_model_v1
Output = 128-dimensional vector
```

### Q6. What is Euclidean distance?

A mathematical measure of the distance between two vectors.

### Q7. Why is smaller distance better?

Because the learned representation is designed so that similar identities occupy nearby regions of the embedding space.

### Q8. What is tolerance?

A threshold used to decide whether the closest candidate is sufficiently similar to be accepted.

### Q9. Is tolerance a confidence percentage?

No.

It is a distance threshold.

### Q10. Why BGR → RGB?

OpenCV loads images in BGR order while the face-recognition pipeline expects RGB.

### Q11. Why resize frames?

To reduce computational cost and improve real-time performance.

### Q12. Why use `np.argmin()`?

To find the index of the smallest face distance.

### Q13. Did you train the model?

No. The project uses a pretrained dlib face-recognition model and generates reference encodings for registered students.

### Q14. Where are encodings stored?

In the current main recognition flow, generated encodings are stored in the Python `known_encodings` list in RAM.

### Q15. What happens when the application closes?

The in-memory encodings are released. On the next run they are generated again from the registered images.

---

# 🧭 Project Learning Roadmap

```text
                 FACE RECOGNITION
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
      COMPUTER VISION          MACHINE LEARNING
          │                         │
          ▼                         ▼
      OpenCV                  Face Encoding
          │                         │
          ▼                         ▼
    Webcam Frames               128D Vector
          │                         │
          └────────────┬────────────┘
                       ▼
                Distance Matching
                       │
                       ▼
                  Recognition
                       │
                       ▼
                  Attendance
```

---

# 🌟 Why This Project Matters

The interesting part of this project is not simply opening a webcam and detecting a face.

The deeper engineering pipeline is:

```text
Raw Pixels
    ↓
Computer Vision
    ↓
Face Localization
    ↓
Neural Representation
    ↓
128D Vector Space
    ↓
Metric Comparison
    ↓
Decision Boundary
    ↓
Identity
    ↓
Attendance Record
```

It combines **computer vision + deep learning + vector mathematics + real-time processing + software engineering** into one practical system.

---

# 🤝 Contributing

Contributions, ideas, improvements, and experiments are welcome.

```bash
git clone <repository-url>

cd Face-Recognition-Attendance-Project-Easy

git checkout -b feature/my-feature

git add .

git commit -m "Add my feature"

git push origin feature/my-feature
```

Then open a Pull Request.

---

# 📜 License

This project is currently intended for educational, learning, and experimentation purposes.

If you intend to distribute the project publicly, add an appropriate open-source license to the repository.

---

# 👨‍💻 Author

<div align="center">

### Likith Naidu

**B.Tech — Computer Science & Engineering (Artificial Intelligence)**

Building projects at the intersection of:

```text
AI × Software Engineering × Computer Vision
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:06b6d4,50:2563eb,100:7c3aed&height=3&section=header" width="70%"/>

</div>

---

# ⭐ If You Find This Project Useful

Give the repository a ⭐ if it helped you learn something about:

```text
Computer Vision
Face Recognition
AI
Python
OpenCV
Machine Learning
```

---

<div align="center">

### 🔥 Built to Learn. Engineered to Understand. 🚀

<br/>

```text
Detect → Encode → Compare → Decide → Record
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:2563eb,100:06b6d4&height=140&section=footer&animation=fadeIn" width="100%"/>

</div>
