import time
import io
import numpy as np
from PyPDF2 import PdfFileReader
from wand.image import Image
from wand.color import Color
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.optimizers import Adam

memo = {}


def getPdfReader(filename):
    reader = memo.get(filename, None)
    if reader is None:
        reader = PdfFileReader(filename, strict=False)
        memo[filename] = reader
    return reader


def _run_convert(filename, page, res=120):
    idx = page + 1
    temp_time = time.time() * 1000
    pdfile = getPdfReader(filename)
    pageObj = pdfile.getPage(page)
    pdf_bytes = io.BytesIO()

    dst_pdf = PdfFileReader()
    dst_pdf.addPage(pageObj)
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)

    img = Image(file=pdf_bytes, resolution=res)
    img.format = 'png'
    img.compression_quality = 90
    img.background_color = Color("white")
    img_path = '%s%d.png' % (filename[:filename.rindex('.')], idx)
    img.save(filename=img_path)
    img.destroy()
    pdf_bytes = None
    dst_pdf = None
    print('convert page %d cost time %d' % (idx, (time.time() * 1000 - temp_time)))


def train_model(X, Y):
    model = Sequential()
    model.add(Flatten(input_shape=(X.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X, Y, epochs=10, batch_size=32, verbose=0)

    return model


def predict_resolution(X, model):
    return model.predict(X)


def preprocess_data(X):
    return X / 1000.0  # Normalize data


if __name__ == '__main__':
    # Simulating data for training the model
    num_pages = 10
    resolutions = np.random.randint(50, 300, num_pages)
    processing_times = np.random.uniform(1, 10, num_pages)

    # Feature matrix (resolutions)
    X = resolutions.reshape(-1, 1)

    # Target variable (processing times)
    Y = processing_times

    # Preprocess data
    X = preprocess_data(X)

    # Train the model
    model = train_model(X, Y)

    # Predict resolution for a new page
    new_resolution = np.array([[150]])  # Example: New page resolution
    new_resolution = preprocess_data(new_resolution)
    predicted_processing_time = predict_resolution(new_resolution, model)[0]
    
    print(f"Predicted Processing Time for Resolution 150: {predicted_processing_time} seconds")

    # Convert a page from PDF
    _run_convert('demo.pdf', 0)
