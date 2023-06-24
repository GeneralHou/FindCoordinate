import cv2
import numpy as np
import json


def extract_coordinates(surface_name, output_dir):
    def shw_img(image, name):
        cv2.namedWindow(name, 0)
        cv2.resizeWindow(name, image.shape[1], image.shape[0]) # w and h
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Load the image
    img = cv2.imread(f'./{output_dir}/{surface_name}_crop.png')

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the image to extract the red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Define the range of red color in HSV (continued)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])

    # Threshold the image to extract the red color (continued)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the two masks
    mask = mask1 + mask2
    shw_img(mask, "mask")

    # coordinates process
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found_coordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 1 and h > 1:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            cv2.circle(img, (cx, cy), 2, (0, 255, 0), -1, cv2.LINE_AA)
            # return to original scale
            found_coordinates.append([round(cx), round(cy)])

    all_coordinates_dict = {x: y for x, y in enumerate(found_coordinates)}
    json_str = json.dumps(all_coordinates_dict, indent=4)
    with open(f'./{output_dir}/coordinates.json', 'w') as json_file:
        json_file.write(json_str)

    cv2.imwrite(f'./{output_dir}/N2_FoundNodes.png', img)

# the next two to lines of code is used to run N1_CornerHarrisPoints.py directly
if __name__ == '__main__':
    extract_coordinates(surface_name='4-000', output_dir='Surface_4-000')