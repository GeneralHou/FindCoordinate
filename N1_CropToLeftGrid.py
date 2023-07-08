import cv2


def shw_img(img, title='default'):
    cv2.namedWindow(title, 0)
    w, h = min(1920, img.shape[1]), min(1080, img.shape[0])
    cv2.resizeWindow(title, w, h) # w and h
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_only_left_frame(img, frame_bounding):
    x, y, w, h = frame_bounding[0], frame_bounding[1], frame_bounding[2], frame_bounding[3]
    x, y, w, h = x+1, y+1, w-2, h-2
    croped_img = img[y:y+h, x:x+w]
    return croped_img


def crop(surface_name):
    output_dir = f'Surface_{surface_name}'
    # make image suitable for with red dots and without red dots
    img_names = [surface_name+x for x in ['_red', '']]
    for img_name in img_names:
        img_path = f'{output_dir}/{img_name}.jpg'
        img = cv2.imread(img_path)
        width, height = img.shape[1] * 15, img.shape[0] * 15
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame_bounding = cv2.boundingRect(contours[2])

        croped_img = crop_only_left_frame(img, frame_bounding)
        cv2.imwrite(f'./{output_dir}/{img_name}_crop.jpg', croped_img)


if __name__ == '__main__':
    crop('1-000')
