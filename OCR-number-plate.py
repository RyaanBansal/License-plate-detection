import cv2
import pytesseract
from PIL import Image
from reportlab.pdfgen import canvas
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\RyaanBansal\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# List of images to process
folder_path = "Sample set"
image_files = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

images = []
for file in image_files:
    img = cv2.imread(os.path.join(folder_path, file))
    if img is not None:
        images.append(img)

# Function to process each image and extract text
def extract_license_plate_text(image):

    # Convert to grayscale
    if image is None:
        print("Error: Image not found or could not be loaded.")
        return "No image detected"
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        roi = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y + h, x:x + w]
                break

        if roi is not None:
            return pytesseract.image_to_string(roi, config='--psm 8').strip()
        else:
            return "License plate not found"

# Function to save multiple extracted texts to a PDF
def save_texts_to_pdf(texts, output_pdf):
    c = canvas.Canvas(output_pdf)
    y_position = 750  # Starting position

    for i, text in enumerate(texts):
        c.drawString(100, y_position, f"Image {i+1}: {text}")
        y_position -= 30  # Move down for the next entry
    
    c.save()

# Process all images
extracted_texts = [extract_license_plate_text(img) for img in images]

# Save results to PDF
save_texts_to_pdf(extracted_texts, "Number plate.pdf")

print("Extracted texts saved to Number plate.pdf")