# ConvoLogix – A Meeting Minutes Generator and Face Attendance System

ConvoLogix is a machine learning and deep learning–based project designed to **automate meeting minutes generation and attendance tracking using face recognition**.  
It addresses inefficiencies of traditional manual note-taking and sign-in sheets by providing a streamlined, accurate, and automated solution.

## Motivation
Manual meeting minutes and attendance processes are:
- Time-consuming and error-prone  
- Prone to missing important points  
- Inefficient for collaboration and data-driven insights  

**ConvoLogix** enhances accuracy, saves time, fosters collaboration, and provides actionable insights by combining **speech-to-text summarization** and **face recognition–based attendance**.

## Objectives
1. **Automated Meeting Summarization**
   - Extract audio from video
   - Convert speech to text
   - Summarize text into concise meeting minutes
   - Identify “who said what” (future scope)

2. **Face Attendance Tracking**
   - Prepare attendee image datasets
   - Detect and recognize faces from meeting video
   - Mark attendance automatically

3. **Integration**
   - Merge minutes + attendance records into a single report
   - Deliver via GUI and email notifications

## Tech Stack

### Languages & Frameworks
- **Python 3.11**
- **TensorFlow / PyTorch**
- **OpenCV (cv2)**
- **Keras**
- **SpaCy, Transformers (Hugging Face)**
- **MoviePy, Spleeter, SpeechRecognition, Pydub**

### Algorithms
- **Convolutional Neural Networks (CNNs)** – Face recognition  
- **Local Binary Pattern Histograms (LBPH)** – Face recognition (98% accuracy achieved)  
- **LexRank, TextRank, Transformers** – Summarization methods

### Tools
- **VS Code, Google Colab**
- **Tkinter** (GUI)
- **SMTP/Email** (to send meeting minutes)
- **NumPy, OS, Datetime** for utilities


## Repository Structure

- `Code/` – Source code for modules  
- `Dataset/` – Face recognition dataset  
- `Extracted Audio/` – Audio separated from video  
- `Minutes of Meeting/` – Generated summaries  
- `Trained Model/` – Pre-trained CNN/LBPH models  
- `Test Accuracy/` – Model evaluation results  
- `Video Screenshots/` – Sample outputs  
- `images/` – Testing images  
- `README.md` – Project documentation  
- `Requirement.txt` – Dependencies  

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shritej21/Convologix-A-Meeting-Minutes-Generator-and-Face-Attendance-System.git
   cd Convologix-A-Meeting-Minutes-Generator-and-Face-Attendance-System

2.   **Install dependencies and run the application**
     ```bash
     pip install -r Requirement.txt
3.  **Run the application**
    ```bash
    python main.py  
   
