# 🚨 Crowd Behavior Analysis System

🟦 Python 3
🟩 OpenCV
🔴 YOLOv4
🟧 Streamlit
🟢 Active

---

## 📌 Overview

This project is a **real-time AI-based crowd monitoring system** that detects human presence and identifies violent or abnormal behavior using computer vision techniques.
It integrates **YOLO object detection**, **motion analysis**, and a **live dashboard** to provide actionable insights.

---

## 🎯 Features

- 👥 Real-time human detection using YOLOv4-tiny  
- ⚠️ Violence detection using motion (optical flow) analysis  
- 🔔 Smart alert system (visual + sound alerts)  
- 📊 Live dashboard visualization using Streamlit  
- 🧾 CSV data logging for analysis  
- 🎥 Processed video output with annotations  

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- YOLOv4-tiny  
- NumPy  
- Streamlit  

---

## 📂 Project Structure
Crowd-Analysis/
│
├── main_with_violence.py
├── config.py
├── dashboard.py
├── violence_detection.py
├── YOLOv4-tiny/
├── report.csv

---

## ⚙️ Installation

pip install opencv-python numpy imutils streamlit pandas

▶️ How to Run

1. Run Detection System:

python main_with_violence.py

2. Run Dashboard (in new terminal):

streamlit run dashboard.py

📊 Output

Real-time detection window
Violence alerts on screen
CSV report (report.csv)
Live dashboard with graphs

🚀 Future Improvements

📧 Email alert system
📱 Mobile notifications
🧠 Deep learning-based action recognition
☁️ Cloud deployment

👤 Author
Mohammad Aadil Raza

⭐ If you found this useful, consider giving it a star!
