# ✨ Handwritten Digit Recognizer – AI Internship Project ✨
**Internship Organization:** Growfinix Technology  
**Internship Domain:** Artificial Intelligence / Deep Learning  
**Project Type:** Task 1 – AI Internship Project  
**Developer:** Ashish Dubey  

---

## 🧠 Project Overview
The **Handwritten Digit Recognizer** is a Deep Learning-based system that identifies handwritten digits (0–9) using **Convolutional Neural Networks (CNNs)** trained on the **MNIST dataset**.  
This project demonstrates how AI models can learn visual patterns like edges, curves, and shapes to interpret human handwriting accurately — bridging the gap between human creativity and machine intelligence.  
Additionally, I built an **interactive Tkinter GUI** where users can draw digits in real time, and the AI model instantly predicts the digit using a trained CNN model. 🎨🤖  

---

## 🧰 Tech Stack & Tools
| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python |
| Core Libraries | NumPy, Pandas, Matplotlib |
| Deep Learning Framework | TensorFlow / Keras |
| Computer Vision Concepts | Feature Extraction, Image Normalization |
| GUI Development | Tkinter, PIL (Python Imaging Library) |
| Dataset | MNIST Handwritten Digits Dataset |

---

## 📂 Dataset
The project uses the **MNIST dataset**, a standard benchmark dataset in the AI community for digit recognition tasks.  
- 📦 **Dataset Source (Kaggle):** [MNIST Handwritten Digits Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
- 📄 The dataset contains **70,000 labeled grayscale images** (60,000 for training and 10,000 for testing), each of size **28×28 pixels**.  

---

## ⚙️ Project Workflow
### 1. Data Preprocessing
- Loaded the MNIST dataset.  
- Normalized pixel values to range `[0, 1]` for faster convergence.  
- Reshaped images to `(28, 28, 1)` for CNN input.  
- One-hot encoded the digit labels (0–9).  

### 2. CNN Model Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### 3. Model Training
- Optimizer: `Adam`  
- Loss Function: `Categorical Crossentropy`  
- Metrics: `Accuracy`  
- Epochs: 10–15  
- Batch Size: 128  
- Trained using the MNIST dataset with validation split.  

### 4. Model Evaluation
- **Training Accuracy:** ~99.3%  
- **Test Accuracy:** ~98.9%  
- Model saved as `.h5` file for reuse.  

### 5. Tkinter GUI Application
A simple GUI built with **Tkinter** lets users draw digits on a canvas. The trained model predicts the digit in real-time.  
**Features:**  
- 🖋️ Draw a digit using the mouse.  
- 🧹 Clear the canvas with one click.  
- ⚡ Get instant AI prediction.  

---

## 🧪 Real-World Applications
| Sector | Use Case |
|--------|-----------|
| 🏦 Banking | Automating cheque processing & handwritten form recognition |
| 📱 Smartphones | Handwriting-to-text input, PIN recognition |
| 📝 Digital Archives | Converting old handwritten records into searchable digital text |
| 🚗 Autonomous Systems | Reading handwritten road signs and research data |

---

## 📈 Results & Insights
✅ **Training Accuracy:** 99.3%  
✅ **Test Accuracy:** 98.9%  
✅ **Deployed as:** Mini AI Desktop Application  

### Key Learnings
- Built and trained CNNs for image classification.  
- Gained hands-on experience in Deep Learning and Computer Vision.  
- Connected theoretical AI concepts with real-world applications.  

---

## 💻 How to Run the Project
### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run the GUI Application
```bash
python app.py
```

---

## 📸 Screenshots
(Add screenshots of your CNN accuracy graph, GUI digit drawer, and prediction output here.)  

| Model Accuracy | GUI Interface | Prediction Output |
|-----------------|----------------|------------------|
| ![Accuracy Graph](assets/accuracy.png) | ![GUI App](assets/gui.png) | ![Result](assets/result.png) |

---

## 🚀 Future Enhancements
- Add model interpretability (Grad-CAM visualization).  
- Deploy as a **web app** using Streamlit or Flask.  
- Extend the system to alphabets and symbols recognition.  
- Create an API endpoint for remote predictions.  

---

## 🙌 Acknowledgements
Special thanks to **Growfinix Technology** for providing this incredible opportunity to work on real-world AI projects as part of my **Artificial Intelligence Internship**.  

> *“AI is not just theory — it’s transforming how we interact with the world every day.”*  

---

## 🧩 Author
**Ashish Dubey**  
AI Intern @ Growfinix Technology  
📧 [your-email@example.com]  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/yourprofile)  
🐙 [GitHub Profile](https://github.com/your-username)  

---

### 📚 Dataset Reference
> [Kaggle: MNIST Handwritten Digits Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

---

⭐ **If you found this project helpful, don’t forget to star the repo!**
