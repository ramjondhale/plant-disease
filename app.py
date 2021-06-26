import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
st.title("Leaf Disease Detection Using Machine Learning")

uploaded_file = st.file_uploader("Choose an image...", type="JPG")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    #image.show()
    st.image(image, caption='Uploaded Image.', width=300)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    data = np.rint(prediction)
    print(data)
    if(data[0][0]==1):
        st.write("Grape___Black_rot")
    if(data[0][1]==1):
        st.write("Grape___Esca_(Black_Measles)")
    if(data[0][2]==1):
        st.write("Grape___healthy")
    if(data[0][3]==1):
        st.write("Grape___Leaf_blight")
