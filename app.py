from flask import Flask, render_template, request, send_file
import cv2
import easyocr
import pandas as pd
import time
import os

app = Flask(__name__)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to detect and recognize number plates
def detect_number_plates(video_path, max_duration=300, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    detected_plates = []
    start_time = time.time()
    elapsed_time = 0

    while elapsed_time < max_duration:
        for _ in range(frame_skip):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use EasyOCR to detect and recognize number plates
        results = reader.readtext(gray)

        # Draw bounding boxes around detected number plates
        for (bbox, text, prob) in results:
            if len(text) >9:  # Check if the length of the detected text is greater than 5
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                # Draw bounding box
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                # Display the recognized text
                cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Add the detected number plate to the list
                detected_plates.append(text)

        # Save the processed frame with bounding boxes
        cv2.imwrite("static/frame.jpg", frame)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

    cap.release()
    # cv2.destroyAllWindows()  # Remove this line

    return detected_plates

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Ensure the 'static' directory exists
            if not os.path.exists("static"):
                os.makedirs("static")

            video_path = "static/temp.mp4"
            file.save(video_path)
            detected_plates = detect_number_plates(video_path)
            df = pd.DataFrame(detected_plates, columns=['Number Plate'])
            df.to_csv("static/detected_number_plates.csv", index=False)
            return render_template("index.html", data=df['Number Plate'].tolist(), download_link="/download")
    return render_template("index.html")

@app.route("/download")
def download():
    return send_file("static/detected_number_plates.csv", as_attachment=True)

@app.route("/results")
def results():
    df = pd.read_csv("static/detected_number_plates.csv")
    data = df['Number Plate'].tolist()
    return render_template("index.html", data=data, download_link="/download")

if __name__ == "__main__":
    app.run(debug=True)
