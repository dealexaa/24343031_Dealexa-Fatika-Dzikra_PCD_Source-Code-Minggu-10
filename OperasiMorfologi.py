#24343031_DEALEXA FATIKA DZIKRA_MINGGU 10

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# ==============================
# LOAD IMAGE
# ==============================
imgA = cv2.imread('citra_A.png', 0)   # dokumen teks
imgB = cv2.imread('citra_B.png')      # koin

# ==============================
# 1. STRUCTURING ELEMENT EXPERIMENT
# ==============================
def experiment_kernels(img):
    sizes = [3,5,7]
    shapes = {
        "RECT": cv2.MORPH_RECT,
        "ELLIPSE": cv2.MORPH_ELLIPSE,
        "CROSS": cv2.MORPH_CROSS
    }

    plt.figure(figsize=(12,8))
    idx = 1

    for s in sizes:
        for name, shape in shapes.items():
            kernel = cv2.getStructuringElement(shape, (s,s))
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            plt.subplot(3,3,idx)
            plt.imshow(result, cmap='gray')
            plt.title(f"{name} {s}x{s}")
            plt.axis('off')
            idx += 1

    plt.suptitle("Eksperimen Structuring Element")
    plt.show()

# ==============================
# 2. BASIC OPERATIONS
# ==============================
def basic_operations(img):
    kernel = np.ones((3,3), np.uint8)

    erosion1 = cv2.erode(img, kernel, iterations=1)
    erosion3 = cv2.erode(img, kernel, iterations=3)

    dilation1 = cv2.dilate(img, kernel, iterations=1)
    dilation3 = cv2.dilate(img, kernel, iterations=3)

    images = [img, erosion1, erosion3, dilation1, dilation3]
    titles = ["Original","Erode x1","Erode x3","Dilate x1","Dilate x3"]

    plt.figure(figsize=(12,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle("Operasi Dasar")
    plt.show()

# ==============================
# 3. ADVANCED MORPHOLOGY
# ==============================
def advanced_morphology(img):
    kernel = np.ones((5,5), np.uint8)

    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    images = [img, gradient, tophat, blackhat]
    titles = ["Original","Gradient","Top Hat","Black Hat"]

    plt.figure(figsize=(10,5))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle("Operasi Morfologi Lanjutan")
    plt.show()

# ==============================
# 4. OCR PREPROCESSING
# ==============================
def ocr_processing(img):
    start = time.time()

    _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_h)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    final = cv2.dilate(closing, kernel_v, iterations=1)

    end = time.time()

    images = [img, binary, opening, closing, final]
    titles = ["Original","Binary","Opening","Closing","Final"]

    plt.figure(figsize=(12,6))
    for i in range(5):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle("OCR Pipeline")
    plt.show()

    return binary, final, (end-start)

# ==============================
# 5. COUNT OBJECT (WATERSHED)
# ==============================
def count_objects(img):
    start = time.time()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opening, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    markers = cv2.watershed(img, markers)

    object_count = len(np.unique(markers)) - 2

    end = time.time()

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(thresh, cmap='gray')
    plt.title("Threshold")

    plt.subplot(1,2,2)
    plt.imshow(markers, cmap='jet')
    plt.title(f"Detected: {object_count}")
    plt.show()

    return object_count, (end-start)

# ==============================
# 6. CONNECTED COMPONENT
# ==============================
def count_components(img):
    num_labels, _ = cv2.connectedComponents(img)
    return num_labels - 1

# ==============================
# 7. TIME ANALYSIS
# ==============================
def measure_time(img):
    kernel = np.ones((3,3), np.uint8)

    t = {}

    start = time.time()
    cv2.erode(img, kernel)
    t['Erosi'] = time.time() - start

    start = time.time()
    cv2.dilate(img, kernel)
    t['Dilasi'] = time.time() - start

    start = time.time()
    cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    t['Opening'] = time.time() - start

    start = time.time()
    cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    t['Closing'] = time.time() - start

    print("\n=== WAKTU KOMPUTASI ===")
    for k,v in t.items():
        print(f"{k}: {v:.6f} detik")

# ==============================
# MAIN
# ==============================
def main():
    print("=== MULAI PROGRAM ===")

    # Kernel experiment
    experiment_kernels(imgA)

    # Basic ops
    basic_operations(imgA)

    # Advanced
    advanced_morphology(imgA)

    # OCR
    binary, final, ocr_time = ocr_processing(imgA)

    before_cc = count_components(binary)
    after_cc = count_components(final)

    ocr_accuracy = (1 - abs(before_cc - after_cc)/before_cc) * 100

    # Counting
    manual_count = 4
    auto_count, count_time = count_objects(imgB)

    counting_accuracy = (auto_count / manual_count) * 100

    # Time analysis
    measure_time(imgA)

    # TABLE
    df = pd.DataFrame({
        "Metode": ["OCR", "Counting"],
        "Sebelum": [before_cc, manual_count],
        "Sesudah": [after_cc, auto_count],
        "Akurasi (%)": [ocr_accuracy, counting_accuracy],
        "Waktu (detik)": [ocr_time, count_time]
    })

    print("\n=== HASIL AKHIR ===")
    print(df)

# ==============================
# RUN
# ==============================
main()