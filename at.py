import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------
# Load TensorFlow model
# -------------------------
tf.config.run_functions_eagerly(True)
model = tf.keras.models.load_model("./digit_recognizer.keras", compile=False)

# -------------------------
# Streamlit App
# -------------------------
st.title("✍️ Handwritten Digit Recognizer (Pure Streamlit)")

# Canvas size
canvas_size = 200
brush_radius = st.slider("Brush size", 5, 25, 15)

# Initialize canvas
if "canvas" not in st.session_state:
    st.session_state.canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255  # white canvas

# Display canvas with matplotlib
fig, ax = plt.subplots(figsize=(2,2))
ax.imshow(st.session_state.canvas, cmap="gray")
ax.axis("off")
st.pyplot(fig)

# Drawing coordinates from user input
x = st.number_input("X coordinate", 0, canvas_size-1, 100)
y = st.number_input("Y coordinate", 0, canvas_size-1, 100)

if st.button("Draw"):
    x_min = max(x - brush_radius, 0)
    x_max = min(x + brush_radius, canvas_size-1)
    y_min = max(y - brush_radius, 0)
    y_max = min(y + brush_radius, canvas_size-1)
    st.session_state.canvas[y_min:y_max, x_min:x_max] = 0  # draw black
    st.experimental_rerun()

if st.button("Clear"):
    st.session_state.canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    st.experimental_rerun()

# Predict button
if st.button("Predict"):
    img = Image.fromarray(st.session_state.canvas)
    img = img.convert("L")
    img_resized = img.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized)
    img_array = np.array(img_inverted) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1).astype("float32")

    prediction = model.predict(img_array, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    st.success(f"Prediction: {digit} (Confidence: {confidence:.2f})")
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# ✅ Force eager execution (fixes tf.data.Dataset error)
tf.config.run_functions_eagerly(True)

# Load trained model (use your saved file)
model = tf.keras.models.load_model("./digit_recognizer.keras")

# Create Tkinter window
window = tk.Tk()
window.title("Handwritten Digit Recognizer")

# Canvas for drawing
canvas = tk.Canvas(window, width=200, height=200, bg="white")
canvas.pack()

# PIL image for drawing
image = Image.new("L", (200, 200), color=255)  # 'L' = grayscale
draw = ImageDraw.Draw(image)

# Function to draw on canvas + image
def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# Prediction label
label = tk.Label(window, text="Draw a digit and click 'Predict'", font=("Arial", 14))
label.pack()

# Function to predict digit
def predict_digit():
    # Resize to 28x28 (MNIST size)
    img_resized = image.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized)  # black on white
    img_array = np.array(img_inverted) / 255.0   # normalize
    img_array = img_array.reshape(1, 28, 28, 1).astype("float32")  # ensure float32

    # ✅ Predict with eager execution
    prediction = model.predict(img_array, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    label.config(text=f"Prediction: {digit} (Confidence: {confidence:.2f})")

# Function to clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 200, 200], fill=255)
    label.config(text="Draw a digit and click 'Predict'")

# Buttons
btn_predict = tk.Button(window, text="Predict", command=predict_digit, font=("Arial", 12))
btn_predict.pack()

btn_clear = tk.Button(window, text="Clear", command=clear_canvas, font=("Arial", 12))
btn_clear.pack()

# Run Tkinter app
window.mainloop()
