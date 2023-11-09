import cv2
from paddleocr import PaddleOCR

class PreprocessCars:
    def __init__(self):
        self.ocr = PaddleOCR(use_gpu=True)

    def read_odometer(self, image):
        results = self.ocr.ocr(image)
        detections = []
        if results:
            for result in results:
                for detection in result:
                    bounding_box = detection[0]
                    text, confidence = detection[1]
                    detections.append((bounding_box, text, confidence))
        return image, detections

    def default(self, image):
        print("\n_____ default processing _____\n")
        return self.read_odometer(image)


    def Mercedes_truck(self, image):
        # Custom preprocessing for Mercedes truck type
        # processed_image = self.some_preprocessing_method_for_mercedes_truck(image)
        print("\n_____Processing for Mercedes truck_____\n")
        # You should return the processed image as well along with the OCR results
        processed_image, detected_texts = self.read_odometer(image)
        return processed_image, detected_texts
    

    def car_type_2(self, image):
        # Custom preprocessing for car type 2
        processed_image = self.some_preprocessing_method_2(image)
        return self.read_odometer(processed_image)

    # ... more car types ...

    # Example preprocessing methods (placeholders)
    def some_preprocessing_method_1(self, image):
        # Implement actual preprocessing for car type 1
        return image

    def some_preprocessing_method_2(self, image):
        # Implement actual preprocessing for car type 2
        return image

# ... add more preprocessing methods as needed ...
