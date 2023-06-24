import cv2


def shw_img(image):
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_only_left_frame(img, frame_bounding):
    x, y, w, h = frame_bounding[0], frame_bounding[1], frame_bounding[2], frame_bounding[3]
    x, y, w, h = x+1, y+1, w-2, h-2
    croped_img = img[y:y+h, x:x+w]
    return croped_img


def crop(surface_name):
    output_dir = 'Surface_' + surface_name
    img = cv2.imread(f'./{output_dir}/{surface_name}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_bounding = cv2.boundingRect(contours[2])

    croped_img = crop_only_left_frame(img, frame_bounding)
    cv2.imwrite(f'./{output_dir}/{surface_name}_crop.png', croped_img)


if __name__ == '__main__':
    crop('4-000')
