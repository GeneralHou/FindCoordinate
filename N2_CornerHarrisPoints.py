# larger the image
# corrorHarris method

import cv2
import numpy as np
from N3_ClickCoordinates import MissingCoorinates
import json


def corner_harris_points(surface_name, output_dir, test_or_not=False):
    # is it a TEST or not ? (the TEST process will show and print the intermediate images)
    test = test_or_not
    # scale factor
    scale_factor = 4
    # image that needed to be processed
    img = cv2.imread(f'./{output_dir}/{surface_name}_crop.png')
    window_h, window_w = img.shape[:2]

    def shw_img(image, name):
        cv2.namedWindow(name, 0)
        cv2.resizeWindow(name, window_w, window_h)
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1, thresh1 = cv2.threshold(gray1, 240, 255, cv2.THRESH_BINARY)
    bgr_img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    dst = cv2.cornerHarris(gray, 4, 5, 0.04)
    ret, thresh = cv2.threshold(dst, 2, 255, cv2.THRESH_BINARY)  # 20240505 adjust:2
    if test: shw_img(thresh,'N20threshold.png')

    # larger the image size
    height, width = thresh.shape[:2]
    thresh = cv2.resize(thresh, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    if test: shw_img(thresh, 'larger_one')

    _, thresh = cv2.threshold(thresh, 90, 255, cv2.THRESH_BINARY)
    if test: shw_img(thresh, 'threshold_again')
    if test: cv2.imwrite(f'./{output_dir}/N20threshold_again.png', thresh)

    # make the sub_image_5.jpg larger
    height, width = img.shape[:2]
    img = cv2.resize(img, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

    # morphology process
    for i in range(10):
        k3 = np.ones((5, 7), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3)

    if test: cv2.imwrite(f'./{output_dir}/N21close.png', mask)

    for i in range(15):
        k1 = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    if test: cv2.imwrite(f'./{output_dir}/N21open.png', mask)

    # coordinates process
    contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found_coordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 1 and h > 1:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1, cv2.LINE_AA)
            # return to original scale
            found_coordinates.append([round(cx / scale_factor), round(cy / scale_factor)])

    missing_coordinates = MissingCoorinates(img).run()
    for ms in missing_coordinates:
        cx = ms[0]
        cy = ms[1]
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1, cv2.LINE_AA)
    # return to original scale
    missing_coordinates = [[round(x / scale_factor), round(y / scale_factor)] for x, y in missing_coordinates[:2]]

    all_coordinates = found_coordinates + missing_coordinates

    print(f'All coordinates:%d, Found crds:%d, Missing crds:%d' % (
    len(all_coordinates), len(found_coordinates), len(missing_coordinates)))
    print(all_coordinates)
    if test: shw_img(img, 'final')

    all_coordinates_dict = {x: y for x, y in enumerate(all_coordinates)}
    json_str = json.dumps(all_coordinates_dict, indent=4)
    with open(f'./{output_dir}/coordinates.json', 'w') as json_file:
        json_file.write(json_str)

    cv2.imwrite(f'./{output_dir}/N23FoundNodes.png', img)


# the next two to lines of code is used to run N1_CornerHarrisPoints.py directly
if __name__ == '__main__':
    corner_harris_points(surface_name='4-D05', output_dir='Surface_4-D05', test_or_not=True)
