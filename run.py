"""
Created on 5/19/2019

@author: sinan

1. Read movie frames
2. Identify the desired window within the frame for movement detection
3. Compute the average pixel intensity of the window
4. Display the average intensity value along with the movie frame
5. Save as new video file
"""

## Import
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy import signal

video_file_name = "vid1.mp4"

print("Processing......", video_file_name)

# Create video object
cap = cv2.VideoCapture(str(video_file_name))

# Horizontal coordinates needed for display purposes
num_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of frames
x_coordinates_list = np.linspace(
    20, 380, int(num_of_frames)
)  # Horizontal axis is divided to equal parts

# Create an empty frame for plotting average pixel intensity
param = np.ones((300, 400, 3), np.uint8) * 255  # White plane
param = cv2.rectangle(
    param, (10, 10), (390, 280), (0, 0, 0), 2
)  # Add a black square around it

# Create a list to keep track of ave-pixel-value per frame
pixel_intensity_vetor = []

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out_name = "output.mp4"
out = cv2.VideoWriter(out_name, fourcc, 30.0, (800, 300))

while cap.isOpened():
    ret, frame = cap.read()

    if ret is True:
        # Resize the frame to make it 480x640
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)

        # Convert from colored to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert from grayscale to black-and-white
        _, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

        # Define a region of interest for the white plastic tray (landmark)
        roi = np.zeros((480, 640), np.uint8)
        roi[240:400, 220:460] = np.ones((160, 240), np.uint8) * 255

        # Bitwise-AND the mask and the original image
        res = cv2.bitwise_and(thresh, thresh, mask=roi)

        # Contours (to find the big white blobs in the image)
        contours, hier = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # print(cv2.contourArea(cnt))
            if 5000 < cv2.contourArea(cnt) < 20000:
                cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)

        # This is to pick the desired white area, which is the plastic tray
        select_cnt = [cnt for cnt in contours if 5000 < cv2.contourArea(cnt) < 20000]
        if len(select_cnt) is not 1:
            print("Problem! Too many/few white areas detected")
            select_cnt = np.array([[300, 300], [300, 300]])
        else:
            xdim, ydim, zdim = select_cnt[0].shape
            select_cnt = select_cnt[0].reshape((xdim, zdim))

        # Coordinates of the landmark (upper left corner)
        min_x = np.min(select_cnt[:, 0])
        min_y = np.min(select_cnt[:, 1])

        # Coordinates of the desired window in which we will observe the movement of rat's paw
        x1, y1 = min_x + 10, min_y - 75
        x2, y2 = min_x + 100, min_y

        # Add a rectangle
        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # img = cv2.line(img, (383,409), (511,409), (0,0,255), 5)

        # Scale the image
        height, width = img.shape[:2]
        img = cv2.resize(
            img,
            (int(width * 0.625), int(height * 0.625)),
            interpolation=cv2.INTER_CUBIC,
        )

        # Plotting the pixel intensity value
        red_area_ave = np.mean(gray[y1:y2, x1:x2])
        y = 280 - (
            red_area_ave * (510 / 255) + 20
        )  # red area average is projected to image's y coordinate
        x = x_coordinates_list[
            int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        ]  # select x coordinate based on the frame number

        param = cv2.line(param, (int(x), 275), (int(x), int(y)), (160, 160, 160), 3)
        param = cv2.circle(param, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(
            param,
            "Average pixel intensity of the red area",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            param, "Time", (180, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )

        # Make them subplots (horizontal concatenation)
        img2 = cv2.hconcat([img, param])

        # Write the frame
        out.write(img2)

        # Display the final frame
        if ret is True:
            cv2.imshow("frame", img2)
            k = cv2.waitKey(25) & 0xFF
            if k == 27:
                break

        # Record the pixel intensity value
        pixel_intensity_vetor.append(red_area_ave)

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Use FINDPEAKS to obtain the number of reaches
piv = np.array(pixel_intensity_vetor)
piv = signal.savgol_filter(piv, 5, 2)  # Smooth the vector
piv = piv - piv.min()  # Normalize the vector 1/2
piv_norm = piv / piv.max()  # Normalize the vector 2/2
peaks, properties = signal.find_peaks(piv_norm, distance=5, prominence=0.2)

# Plotting
plt.plot(piv_norm)
plt.plot(peaks, piv_norm[peaks], "o")
plt.show()
plt.pause(0.5)
plt.close()

# Report
print("Rat performed {} reaches.".format(len(peaks)))
