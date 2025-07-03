# app.py
import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("siamese_no_lambda.h5")

def predict_signature(img1, img2):
    img1 = img1.resize((224,224)).convert("L")
    img2 = img2.resize((224,224)).convert("L")

    arr1 = np.array(img1, dtype=np.float32) / 255.0
    arr2 = np.array(img2, dtype=np.float32) / 255.0

    arr1 = arr1.reshape(1,224,224,1)
    arr2 = arr2.reshape(1,224,224,1)

    score = model.predict([arr1, arr2])[0][0]
    result = "Genuine" if score >= 0.5 else "Forged"
    return f"{result} (Score: {score:.2f})"

gr.Interface(
    fn=predict_signature,
    inputs=[
        gr.Image(type="pil", image_mode='L', label="Signature 1"),
        gr.Image(type="pil", image_mode='L', label="Signature 2")
    ],
    outputs="text",
    title="Signature Verification"
).launch()
