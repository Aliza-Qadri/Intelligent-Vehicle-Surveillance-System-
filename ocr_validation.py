# Function to check OCR working
def validate_ocr(image, detected_text):
    print("\n--- OCR Validation ---")
    if image is not None and detected_text:
        print(f"Detected Text: {detected_text}")
        print("Image is successfully processed by OCR.")
    else:
        print("OCR failed to detect text.")
    print("----------------------\n")
