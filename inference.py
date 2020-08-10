import argparse
import numpy as np

import cv2
from tensorflow.keras.models import load_model


def main(args):
    model = load_model(args.model)

    # prevents openCL usage and unnecessary logging messages
    # cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {
        0: "Angry",
        1: "Disgusted",
        2: "Scared",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised",
    }

    # start the webcam feed
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() == False:
        print("Error reading video file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter(
        "sample_output.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        cap.get(cv2.CAP_PROP_FPS),
        size,
    )
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=2
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2
            )
            roi_gray = gray[y : y + h, x : x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
            )
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(
                frame,
                emotion_dict[maxindex],
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        result.write(cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC))
        # cv2.imshow(
        #     "Video",
        #     cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC),
        # )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--video", help="Path for input video.")
    args.add_argument("--model", help="Keras model path.")

    main(args.parse_args())
