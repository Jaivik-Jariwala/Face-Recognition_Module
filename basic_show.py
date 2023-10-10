import cv2
from skimage.metrics import structural_similarity as ssim

# Load the reference image from the directory
reference_image = cv2.imread('jaivik.jpeg')
reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Get the dimensions of the reference image
ref_height, ref_width = reference_image_gray.shape

# Initialize the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Define a threshold for similarity
threshold = 0.95  # Adjust this value as needed

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to match the dimensions of the reference image
    frame = cv2.resize(frame, (ref_width, ref_height))

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM between the frame and the reference image
    ssim_score = ssim(frame_gray, reference_image_gray)

    # Check if the frame is similar to the reference image
    if ssim_score > threshold:
        cv2.putText(frame, "Match", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Match", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
