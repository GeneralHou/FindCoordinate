# this .py file does not run directly and it will be imported by other files
import cv2


class MissingCoorinates:
    def __init__(self, img):
        self.img = img
        self.missing_coordinates = []  # used to store the missing coordinates

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', self.callback_function)

    # define the callback function (the callback has its default expression, and it must be obeyed)
    def callback_function(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.missing_coordinates.append([x, y])
            print(f'The clicked coordinate value is [{x}, {y}]')

    def run(self):
        while True:
            cv2.imshow('image', self.img)
            key = cv2.waitKey()
            if key == 27:  # press to 'ESC' to withdraw
                break
        cv2.destroyAllWindows()
        return self.missing_coordinates
