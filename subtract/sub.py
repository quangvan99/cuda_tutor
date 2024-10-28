import cv2
import numpy as np
import time

def main():
    # Load the images
    image1 = cv2.imread('im/t1.jpg')
    image2 = cv2.imread('im/t2.jpg')

    if image1 is None or image2 is None:
        print("Error: Images not found")
        return

    # Resize images
    size = (10000, 10000)
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)

    # Subtract images multiple times and measure the time
    durations = []
    for _ in range(100):
        start_time = time.time()
        result = cv2.subtract(image1, image2)
        end_time = time.time()
        duration = (end_time - start_time)  # Convert to milliseconds
        durations.append(duration)
        print(f"{duration:.4f}s")

    # Optionally, calculate and print average duration
    average_duration = sum(durations) / len(durations)
    print(f"Average Duration: {average_duration:.4f}s")

if __name__ == "__main__":
    main()
