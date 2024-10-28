import cv2
import numpy as np
import time

def main():
    # Load the image in grayscale
    img = cv2.imread("im/t1.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (10000, 10000))

    durations = []
    for i in range(10):
        start_time = time.time()

        # Sobel operations
        ddepth = cv2.CV_16S
        ksize = 3

        # Gradient X
        grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)

        # Gradient Y
        grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # Total Gradient (approximate)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        end_time = time.time()
        duration = (end_time - start_time)
        durations.append(duration)
        print(f"{duration:.4f}s")

    # Optionally, calculate and print average duration
    average_duration = np.mean(durations)
    print(f"Average Duration: {average_duration:.4f}s")

if __name__ == "__main__":
    main()
