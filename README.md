## Code Highlights

### Accuracy Improvements

- **Lower Confidence Threshold**:
    ```python
    min_confidence = 0.3
    ```

- **Adjusted NMS Threshold**:
    ```python
    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.3)
    ```

- **Input Blob Preprocessing**:
    ```python
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    ```

- **Print Detected Objects**:
    ```python
    print(f"Detected: {classes[class_index]}, Confidence: {confidence:.2f}")
    ```

### Final Code Adjustments

- Iterate over `indices` directly.
- Handle empty detections properly.
- Adjust `min_confidence` and NMS parameters based on the output.
