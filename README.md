# 🖐️ Waiter Calling System using YOLOv8 Pose Estimation

## 📌 Overview
This project implements an **automatic waiter-calling system** for classrooms, offices, or meeting rooms using **YOLOv8 object detection and pose estimation**.

The system:
- **Detects desks automatically** from classroom video.
- **Assigns desk labels & student names**.
- **Monitors raised hands** using YOLOv8 Pose.
- Displays the **name of the person raising their hand** in the output video.

---

## 🔹 Pipeline

### 1. Desk Detection & Labeling
- Use YOLOv8 to detect people in classroom video.
- Enhance frame contrast for better detection.
- Remove very small or overlapping boxes.
- Sort desk regions **left → right** and label as `A, B, C ...`.
- Insert a special desk **“X” before desk F**.
- Save final desk coordinates + labels.

### 2. Hand Raise Detection
- Apply YOLOv8 Pose to extract body keypoints.
- Slightly expand desk bounding boxes for accuracy.
- Map each desk label → person’s name.
- For each detected person:
  - Check if a hand is **above shoulder level** with confidence.
  - Match detection to correct desk region.
  - Overlay **student name** on video if hand is raised.
- Save annotated video as `final_handraise_output.mp4`.

---

## 🎥 Input / Output

- **Input**
  - Classroom video (`desk_video.mp4`)
  - Image with desk names for reference
  - Mapping of desk labels → student names

- **Output**
  - Processed video (`final_handraise_output.mp4`) showing:
    - Labeled desk regions
    - Student names when they raise their hands

---

## 🚀 Features
- ✅ Automatic desk region detection & labeling
- ✅ Pose-based hand raise detection
- ✅ Student name overlay for identification
- ✅ Handles overlapping/small detections
- ✅ Exports annotated video

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/<your-username>/waiter-calling.git
cd waiter-calling

# Install dependencies
pip install ultralytics opencv-python-headless pandas numpy
```

---

## ▶️ Usage

1. Place your classroom video as `desk_video.mp4`.
2. Open and run the notebook:

```bash
jupyter notebook waiter_calling_Anita_Tasnim.ipynb
```

3. The final annotated video will be saved as:

```
final_handraise_output.mp4
```

---

## 📊 Results

- Desks automatically labeled as `A, B, C...` (+ special `X`).
- Student names displayed when a hand is raised.

📸 Example (Frame Extract):

![Desk Regions Example](image_with_names.png)

---

## 🏗️ Future Work
- Real-time webcam integration
- Multi-camera classroom support
- Web app deployment

---

## 🙌 Credits
Developed by **Anita Tasnim**  
Built with [YOLOv8](https://github.com/ultralytics/ultralytics) + OpenCV
