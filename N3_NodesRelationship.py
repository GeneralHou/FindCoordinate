import cv2
import json
import numpy as np
import random


def shw_img(img, title='default'):
    cv2.namedWindow(title, 0)
    w, h = min(1920, img.shape[1]), min(1080, img.shape[0])
    cv2.resizeWindow(title, w, h) # w and h
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def is_red(img, coordinate):
    red = [0,0,255]
    point_pixel = img[coordinate[1], coordinate[0]]
    judge = np.array_equal(point_pixel, red)
    return judge

def is_black(img, coordinate):
    black = [0,0,0]
    point_pixel = img[coordinate[1], coordinate[0]]
    judge = np.array_equal(point_pixel, black)
    return judge

# judge a coordinate is in a image or not
def exceed_bound(img, coordinate):
    img_h, img_w = img.shape[:2]
    judge = coordinate[0] >= img_w or coordinate[0] < 0 or \
            coordinate[1] >= img_h or coordinate[1] < 0
    return judge

def find_red_bound(img, coordinate):
    red_bound = {'L':[-1,0], 'R':[1,0], 'T':[0,-1], 'B':[0,1]}
    for k, move_one_pixel in red_bound.items():
        current_coord = coordinate
        while is_red(img, current_coord) == True:
            found_bound = current_coord
            current_coord = [x+y for x,y in zip(current_coord, move_one_pixel)]
            # if current coordinate exceed boundary of image, break while in advance
            if exceed_bound(img, current_coord) == True: break
        red_bound[k] = found_bound
    return red_bound

# crop rectangle boundary of red dot(node) as image to further detect black lines
def crop_red_rectangle(img, rL, rR, rT, rB):
    expand_factor = 0.5
    expand_length = max((rR-rL)*expand_factor, (rB-rT)*expand_factor)
    x1, y1 = rL-expand_length, rT-expand_length
    x2, y2 = rR+expand_length, rB+expand_length
    # avoid red rectangle exceed the img boundary and force float to int
    x1, y1 = int(x1), int(y1)  # left top coordinate
    x2, y2 = int(x2), int(y2)
    red_rect_img = img[y1:y2, x1:x2]
    anchor = [x1, y1]
    return red_rect_img, anchor

# find out black lines connected to red dot in rectangle boundary
def black_lines_corresponding_centers(img, anchor):
    # when running the code, a problem occurred: when red dots are not big enough
    # the black lines may connects to each other, making cv2.connectedComponentsWithStats not work
    # to fix this problem, we erode the black pixels to force lines not connect 
    def erosion(img):
        big_kernel = np.ones((5,5), np.uint8)
        img = cv2.erode(img, big_kernel, iterations=2)
        small_kernel = np.ones((2,2), np.uint8)
        img = cv2.erode(img, small_kernel, iterations=2)
        return img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # !!!!!'10' below is a changeable parameter
    # !!!!! it is to extract black pixel but avoid red pixel (under grayscale)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    # use function define above
    erosion_img = erosion(thresh)
    # Find the black pixel clusters
    _, _, _, centroids = cv2.connectedComponentsWithStats(erosion_img, 8, cv2.CV_32S)
    # function below will return 4 results and we just need the final one
    local_center_list = [[int(x[0]), int(x[1])] for x in centroids[1:]]  # centroids[0] is background
    # adjust the coordinates from local to global
    global_center_list = [[x[0]+anchor[0], x[1]+anchor[1]] for x in local_center_list]
    return global_center_list

# try to find out other red dots from a given start point list and the starting red dot
def find_other_red_dots(img, start_point_list, origin_red_dot):
    # final result in this function(use to store coordinates of the red dots we have found)
    found_red_dots_coords = []
    
    def calculate_dist(p1, p2):  # to make calculate dist in while more intuitive
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    for start_point in start_point_list:
        # define current point and distance
        current_point = start_point
        current_dist = calculate_dist(start_point, origin_red_dot)

        # Define the directions to search
        # it is extremely import to add the first four directions, otherwise, it will bump into infinite loop
        directions = [[1,1], [1,-1], [-1,-1], [-1,1],[0, 1], [0, -1], [1, 0], [-1, 0]]

        while True:
            random_indices = random.sample(range(8), 8)  # randomly choose from list, make sure every direction has opportunity
            for index in random_indices:
                direct = directions[index]
                next_point = [current_point[0]+direct[0], current_point[1]+direct[1]]
                next_dist = calculate_dist(next_point, origin_red_dot)
                if exceed_bound(img, next_point):  # whether exceed the boundary of the image or not
                    break  # should execute first(otherwise, code will raise error when out of image boundary)
                elif is_red(img, next_point):  # the result we want
                    found_red_dots_coords.append(next_point)
                    break
                elif is_black(img, next_point) and next_dist >= current_dist:
                    current_point, current_dist = next_point, next_dist
                    break
                else:
                    # if next_point is not black or is black but distance < current
                    # continue to search next direction
                    continue
            if exceed_bound(img, next_point) or is_red(img, next_point): 
                break  # break from 'while' after 'break' from 'if' above
    return found_red_dots_coords

# after we have found child red dots, we continue to find out their corresponding serial numbers
def serial_number_pairs(mom_dot_key, child_red_dots_coord, coordinates):
    paired_result = []
    for child_coord in child_red_dots_coord:
        dist = float('inf')
        for k, v in coordinates.items():
            if k == mom_dot_key: continue # search dict of coordinates not including mother key and coordinate
            new_dist = np.sqrt((v[0] - child_coord[0])**2 + (v[1] - child_coord[1])**2)
            if new_dist <= dist:
                dist = new_dist
                temp_k = k  # store the key corresponding to current minimal distance
        paired_result.append(sorted([temp_k, mom_dot_key]))
    return paired_result

def relation_in_image(img, coordinates, test_mode=False, test_n=0):
    adjacency_relation = []
    for n, coord in coordinates.items():  # n: serial number
        if test_mode: n, coord = test_n, coordinates[test_n]
        print(f'Processing Node {n}: with coord {coord}')
        '''search two direction(x and y) to find out the boundary of red dot(node)'''
        red_bound = find_red_bound(img, coord)

        '''find out how many black lines connected to the red dot(node)'''
        rL, rR = red_bound['L'][0], red_bound['R'][0]  # 'left top' of the boundary rectangle(of the red dot)
        rT, rB = red_bound['T'][1], red_bound['B'][1]  # 'right bottom' of the boundary rectangle
        # crop the red boundary rectangle as an image
        red_rect_img, anchor = crop_red_rectangle(img, rL, rR, rT, rB)
        # get the list of black lines(and centers)
        center_list = black_lines_corresponding_centers(red_rect_img, anchor)

        '''search other red dots(nodes)'''
        found_red_dots = find_other_red_dots(img, center_list, coord)  # the coordinates of the found
        # find out serial numbers of our found red dots
        serial_pairs = serial_number_pairs(n, found_red_dots, coordinates)
        # add to the final list
        print(f'Processing Node {n}: Finished')
        adjacency_relation += serial_pairs
        if test_mode: break

    return adjacency_relation

def nodes_relationship(surface_name, test_mode=False, test_n=0):
    '''''''''''''PREPARE FOR MAIN'''''''''''''''
    '''DEFINE FILE PATHS'''
    output_dir = 'Surface_' + surface_name
    # image path
    # img_path = f'./{output_dir}/{surface_name}_nored_crop.png'
    img_path = f'./{output_dir}/{surface_name}_crop.jpg'
    # coordinates path
    coord_path = f'./{output_dir}/coordinates.json'

    '''LOAD DATA'''
    # load coordinates data
    with open(coord_path, 'r') as f:
        result = json.load(f)
    coordinates = {int(k): v for k, v in result.items()}

    # load image and preprocess it
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img[img < 230] = 0  # make it darker(reason: some lines are gray and not that clear)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Define the new dimensions

    for _, v in coordinates.items():
        cv2.circle(img, tuple(v), radius=45, color=(0,0,255), thickness=-1)
    if test_mode: cv2.imwrite(f'./{output_dir}/N3_{surface_name}_test.jpg', img)
    print('Red dots can be set to smaller or bigger by adjusting source code.')
    print('This two parameter can be adjusted: radius and thickness.')
    shw_img(img, 'Please check the image is perfect enough to be process(from N3.py)')
    input('Press ENTER to continue(ctrl+c to quit) >>>>>>')

    '''''''''''''MAIN PART OF THIS FUNCTION'''''''''''''''
    adjacency_relation = relation_in_image(img, coordinates, test_mode, test_n)

    # delete duplicated adjacency relation in the list
    adjacency_relation = [list(x) for x in set(tuple(x) for x in adjacency_relation)]
    
    # save the final result
    with open(f'./{output_dir}/adjacency_relation.json', 'w') as fp:
        json.dump(adjacency_relation, fp, indent=4)


if __name__ == '__main__':
    nodes_relationship(surface_name='1-000', test_mode=False, test_n=303)

