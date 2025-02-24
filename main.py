import io
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Response
import uvicorn
from typing import List
import os

app = FastAPI()

# Variabel global untuk menyimpan interpreter TFLite & label map
interpreter = None
input_details = None
output_details = None
label_map = []

@app.on_event("startup")
def load_model():
    """
    Fungsi ini akan dipanggil otomatis saat aplikasi FastAPI mulai.
    Digunakan untuk memuat model TFLite dan label map.
    """
    global interpreter, input_details, output_details, label_map

    # Path model TFLite
    model_path = os.path.join("model", "detect.tflite")

    # Inisialisasi TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load label map
    # Misal labelmap.txt berisi:
    #   black-line
    #   healthy
    #   onychomycosis
    #   psoriasis
    labelmap_path = os.path.join("model", "labelmap.txt")
    with open(labelmap_path, "r") as f:
        lines = f.read().strip().split("\n")
        label_map = [line.strip() for line in lines if line.strip()]

    # Simpan di global
    globals()["interpreter"] = interpreter
    globals()["input_details"] = input_details
    globals()["output_details"] = output_details
    globals()["label_map"] = label_map

    print("TFLite model & label map loaded successfully!")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint untuk menerima upload gambar,
    melakukan object detection dengan TFLite,
    dan mengembalikan gambar hasil deteksi (bounding box + label).
    """
    try:
        # Baca file gambar yang diupload
        image_data = await file.read()

        # Ubah menjadi PIL image
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        np_image = np.array(pil_image)  # shape: (H, W, 3)

        # Ambil info input model
        # Biasanya [1, height, width, 3]
        in_height = input_details[0]["shape"][1]
        in_width = input_details[0]["shape"][2]

        # Resize ke ukuran input model
        resized = cv2.resize(np_image, (in_width, in_height))

        # Buat batch [1, height, width, 3]
        input_data = np.expand_dims(resized, axis=0)

        # Cek tipe data yang diharapkan (UINT8 atau FLOAT32)
        input_dtype = input_details[0]["dtype"]  # ex: np.uint8, np.float32
        if input_dtype == np.float32:
            # Normalisasi 0..1 jika diperlukan
            input_data = input_data.astype(np.float32) / 255.0
        else:
            # Jika model uint8, cukup cast ke uint8
            input_data = input_data.astype(np.uint8)

        # Set tensor input
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # Ambil output
        # Format umum: [boxes, classes, scores, numDetections]
        # Urutan tergantung model TFLite. Cek output_details.
        boxes = interpreter.get_tensor(output_details[0]["index"])       # shape: [1, N, 4]
        classes = interpreter.get_tensor(output_details[1]["index"])     # shape: [1, N]
        scores = interpreter.get_tensor(output_details[2]["index"])      # shape: [1, N]
        num_detections = interpreter.get_tensor(output_details[3]["index"])[0]  # shape: [1], int

        # boxes -> [ymin, xmin, ymax, xmax] (normalisasi 0..1)
        # classes -> class index
        # scores -> confidence
        # num_detections -> int

        # Gambar bounding box pada np_image (ukuran asli)
        h_orig, w_orig, _ = np_image.shape
        boxes = boxes[0]    # shape: [N, 4]
        classes = classes[0]
        scores = scores[0]

        threshold = 0.5  # confidence threshold
        for i in range(int(num_detections)):
            score = scores[i]
            if score >= threshold:
                # Dapatkan koordinat bounding box (normalisasi)
                ymin, xmin, ymax, xmax = boxes[i]
                (ymin, xmin, ymax, xmax) = (
                    int(ymin * h_orig),
                    int(xmin * w_orig),
                    int(ymax * h_orig),
                    int(xmax * w_orig)
                )

                class_id = int(classes[i])
                # Pastikan class_id ada di range label_map
                label = label_map[class_id] if class_id < len(label_map) else "unknown"

                # Gambar kotak
                cv2.rectangle(np_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Teks label + skor
                cv2.putText(
                    np_image,
                    f"{label} {score*100:.2f}%",
                    (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Konversi np_image kembali ke JPEG
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
