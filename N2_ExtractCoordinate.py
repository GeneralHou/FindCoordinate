import cv2
import numpy as np
import json


def extract_coordinates(surface_name):
    output_dir = 'Surface_' + surface_name
    def shw_img(img, title='default'):
        cv2.namedWindow(title, 0)
        cv2.resizeWindow(title, img.shape[1], img.shape[0]) # w and h
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Load the image
    img = cv2.imread(f'./{output_dir}/{surface_name}_red_crop.jpg')

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
    shw_img(mask, "mask(from N2.py)")

    print('If some nodes are too close and connected, You can use morphology operation: OPEN')
    print("You can adjust 'kernel'in the source code")
    # adjust the kernel below may help, but it can not solve the problem thoroughly
    # to avoid this problem, we should not use nodes that are too close to each other
    input("Press 'ENTER' to continue >>>>>>>")

    # morphology operation: open
    kernel = np.ones((3,3), np.uint8)  # kernel size ##########################################
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    shw_img(mask, "after_open(from N2.py)")

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

    cv2.imwrite(f'./{output_dir}/N2_FoundNodes.jpg', img)

# the next two to lines of code is used to run N1_CornerHarrisPoints.py directly
if __name__ == '__main__':
    extract_coordinates(surface_name='1-000')