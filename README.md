# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

```
# ============================================================
# SUNGLASSES IMAGE PROCESSING PROJECT
# Using OpenCV, NumPy and Matplotlib
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# 1. LOAD THE FACE IMAGE
# ------------------------------------------------------------

face_path = "myphoto.jpeg"
glasses_path = "sunglass.png"

face_bgr = cv2.imread(face_path)

if face_bgr is None:
    raise FileNotFoundError(
        f"Cannot find {face_path}. "
        "Place it in the same folder as this notebook."
    )

# Convert BGR to RGB for displaying with Matplotlib
face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

print("Face image loaded successfully.")
print("Face dimensions:", face.shape)

# ------------------------------------------------------------
# 2. LOAD THE SUNGLASSES PNG WITH TRANSPARENCY
# ------------------------------------------------------------

glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

if glasses is None:
    raise FileNotFoundError(
        f"Cannot find {glasses_path}. "
        "Place it in the same folder as this notebook."
    )

if len(glasses.shape) != 3 or glasses.shape[2] != 4:
    raise ValueError(
        "The sunglasses image must be a transparent PNG "
        "with four channels (BGRA)."
    )

print("Sunglasses image loaded successfully.")
print("Sunglasses dimensions:", glasses.shape)

# ------------------------------------------------------------
# 3. DISPLAY THE ORIGINAL IMAGES
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(face)
plt.title("Original Face")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(glasses, cv2.COLOR_BGRA2RGBA))
plt.title("Sunglasses PNG")
plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. DETECT THE FACE AND EYE REGION
# ------------------------------------------------------------

gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60)
)

# ------------------------------------------------------------
# 5. POSITION AND BLEND THE SUNGLASSES
# ------------------------------------------------------------

def overlay_sunglasses(image, glasses_png, x, y, width):
    """
    Resize transparent sunglasses and blend them onto an RGB image.
    x and y are the top-left coordinates of the overlay.
    """

    result = image.copy()
    H, W = result.shape[:2]

    width = max(1, int(width))
    height = max(
        1,
        int(glasses_png.shape[0] * width / glasses_png.shape[1])
    )

    overlay = cv2.resize(
        glasses_png,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # Clip overlay coordinates to the image boundaries
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + width)
    y2 = min(H, y + height)

    if x1 >= x2 or y1 >= y2:
        return result

    # Match the corresponding overlay crop
    gx1 = x1 - x
    gy1 = y1 - y
    gx2 = gx1 + (x2 - x1)
    gy2 = gy1 + (y2 - y1)

    cropped = overlay[gy1:gy2, gx1:gx2]

    # Convert sunglasses colors from BGR to RGB
    glass_rgb = cv2.cvtColor(
        cropped[:, :, :3],
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    # Preserve alpha transparency
    alpha = (
        cropped[:, :, 3:4].astype(np.float32) / 255.0
    )

    # Alpha blending
    roi = result[y1:y2, x1:x2].astype(np.float32)

    blended = roi * (1.0 - alpha) + glass_rgb * alpha

    result[y1:y2, x1:x2] = np.clip(
        blended, 0, 255
    ).astype(np.uint8)

    return result


# Start with an unchanged copy of the face
result = face.copy()

if len(faces) > 0:

    # Use the largest detected face
    fx, fy, fw, fh = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]

    # Detect eyes within the detected face
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(10, fw // 12), max(10, fh // 12))
    )

    # Keep eyes in the upper part of the face
    eyes = [
        (ex, ey, ew, eh)
        for ex, ey, ew, eh in eyes
        if ey < fh * 0.65
    ]

    if len(eyes) >= 2:

        # Select the two detected eyes that are farthest apart
        # horizontally
        eyes = sorted(
            eyes,
            key=lambda eye: eye[0] + eye[2] / 2
        )

        left_eye = eyes[0]
        right_eye = eyes[-1]

        left_cx = fx + left_eye[0] + left_eye[2] / 2
        right_cx = fx + right_eye[0] + right_eye[2] / 2

        eye_y = (
            fy + left_eye[1] + left_eye[3] / 2 +
            fy + right_eye[1] + right_eye[3] / 2
        ) / 2

        # Estimate glasses width from the distance between eyes
        eye_distance = right_cx - left_cx
        glasses_width = int(eye_distance * 2.15)

        # Center the glasses between the eyes
        glasses_x = int(
            (left_cx + right_cx) / 2 - glasses_width / 2
        )

        # Adjust vertical placement slightly above eye centers
        glasses_y = int(eye_y - glasses_width *
                        glasses.shape[0] /
                        glasses.shape[1] * 0.48)

        result = overlay_sunglasses(
            face,
            glasses,
            glasses_x,
            glasses_y,
            glasses_width
        )

        print("Sunglasses positioned using detected eyes.")

    else:
```
## OUTPUT
<img width="561" height="668" alt="image" src="https://github.com/user-attachments/assets/95bc96fb-5581-464c-b925-c28c752eaa0d" />


<img width="635" height="306" alt="image" src="https://github.com/user-attachments/assets/d09525b7-82d4-4aa0-b1ad-afc9b8a0ba4c" />


<img width="665" height="746" alt="image" src="https://github.com/user-attachments/assets/c8f1daa0-ec5c-4e7c-9d26-375031faf1d3" />



<img width="1137" height="287" alt="image" src="https://github.com/user-attachments/assets/bfe896e5-ea11-4bb9-bbf3-eacd757213ad" />


<img width="713" height="726" alt="image" src="https://github.com/user-attachments/assets/44ce613d-325e-4907-8277-8c75f689c06f" />



<img width="1143" height="602" alt="image" src="https://github.com/user-attachments/assets/7d4b413e-baaf-4237-b285-faa6b8c197e7" />


<img width="1147" height="712" alt="image" src="https://github.com/user-attachments/assets/a39185ed-3949-4e13-866c-bdab0bf976c0" />
