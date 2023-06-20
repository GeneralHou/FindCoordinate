import cv2
import numpy as np
from PIL import Image
import math
import json
from skimage import morphology


def new_sub_img_size(img_size_, expand_, factor_):
    sz = np.array(img_size_)
    exp = np.array(expand_)
    fct = np.array(factor_)
    expanded_img = sz + exp * fct
    # img pixel must be int, therefore we use function 'round' to change 'float' to 'int'
    expanded_img = [round(x) for x in expanded_img]
    return expanded_img


def get_ij_sub_img(img_path, nh, nw, i, j, expand_factor):
    image = Image.open(img_path)

    # the size of sub_image
    width, height = image.size
    sub_width = width // nw
    sub_height = height // nh

    # expand distance
    exp_w = sub_width * expand_factor
    exp_h = sub_height * expand_factor
    expand = [exp_w, exp_h, exp_w, exp_h]

    # the default crop
    left = j * sub_width
    top = i * sub_height
    right = left + sub_width
    bottom = top + sub_height
    img_size = [left, top, right, bottom]

    # calculate order(tl,tr,br,bl): first_4_corners, then_4_sides, finally_middle_area
    # expand direction: left_top_right_bottom; expand(1) un_expand(0)

    # 4 corners: (0011, -1001, -1-100, 0-110)
    if i == 0 and j == 0:
        factor = [0, 0, 1, 1]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)
    elif i == 0 and j == nw-1:
        factor = [-1, 0, 0, 1]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)
    elif i == nh-1 and j == nw-1:
        factor = [-1, -1, 0, 0]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)
    elif i == nh-1 and j == 0:
        factor = [0, -1, 1, 0]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)

    # 4 sides: (-1011, -1-101, -1-110, 0-111)
    elif i == 0 and 0 < j < nw-1:
        factor = [-1, 0, 1, 1]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)
    elif 0 < i < nh-1 and j == nw-1:
        factor = [-1, -1, 0, 1]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)
    elif i == nh-1 and 0 < j < nw-1:
        factor = [-1, -1, 1, 0]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)
    elif 0 < i < nh-1 and j == 0:
        factor = [0, -1, 1, 1]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)

    # middle area: (-1-111)
    else:
        factor = [-1, -1, 1, 1]
        expanded_img_size = new_sub_img_size(img_size, expand, factor)

    # crop sub_image
    sub_image = image.crop(expanded_img_size[:4])

    return sub_image, expanded_img_size


# use to calculate the coordinates(nodes) in the area of sub image
def coordinates_in_sub_img(sub_size_, coordinates):
    x1, x2 = sub_size_[0], sub_size_[2]
    y1, y2 = sub_size_[1], sub_size_[3]
    coordinates_in_sub_ = {}
    for k, v in coordinates.items():
        in_x_or_not = x1 <= v[0] <= x2 or x2 <= v[0] <= x1
        in_y_or_not = y1 <= v[1] <= y2 or y2 <= v[1] <= y1
        if in_x_or_not and in_y_or_not:
            coordinates_in_sub_[k] = v
    return coordinates_in_sub_


def calculate_angle(coordinates, a_random, b, c):  # A_random, B and C is the key of dictionary
    # coordinates of Three points
    xa, ya = coordinates[a_random][:2]
    xb, yb = coordinates[b][:2]
    xc, yc = coordinates[c][:2]
    # define vector: AB and AC
    v_ab = np.array([xb - xa, yb - ya])
    v_ac = np.array([xc - xa, yc - ya])
    # the angle between two AB and AC
    cos_angle = np.dot(v_ab, v_ac) / (np.linalg.norm(v_ab) * np.linalg.norm(v_ac))
    angle = np.arccos(cos_angle)
    # radian to angle
    angle = np.degrees(angle)
    return angle


# this function is used in 'TEST' mode
def show_coordinate_keys(coordinates, img_path, surface_name, output_dir):
    img = cv2.imread(img_path, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (90, 205, 106), (240, 160, 32),
             (42, 42, 128), (32, 240, 160), (61, 139, 72), (142, 35, 107)]
    i = 0
    h, w = img.shape[:2]
    for k, v in coordinates.items():
        xx, yy = v[:2]
        xx = xx if xx < w else xx-w
        yy = yy if yy < h else yy-h
        cv2.putText(img, str(k), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[i], 1)
        cv2.circle(img, tuple([x for x in v[:2]]), 1, color[i], 2)
        if i < len(color)-1:
            i += 1
        else:
            i = 0
    cv2.namedWindow('Coordinate_keys', cv2.WINDOW_NORMAL)
    cv2.imshow('Coordinate_keys', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'./{output_dir}/{surface_name}_crop_with_keys.png', img)
    return img


# this function is used in 'TEST' mode
def draw_line(coordinates, key1, key2, img):
    cv2.line(img, tuple([x for x in coordinates[key1]]), tuple([x for x in coordinates[key2]]), (0, 0, 255), 2)
    cv2.namedWindow('draw_line', cv2.WINDOW_NORMAL)
    cv2.imshow('draw_line', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def avoid_point_in_two_points(coordinates, img_path, surface_name, output_dir):
    if TEST: img = show_coordinate_keys(coordinates, img_path, surface_name, output_dir)
    keys = list(coordinates.keys())
    length = len(keys)
    lines = []  # used to store paired points
    for i in range(length - 1):
        for j in range(i + 1, length):
            key1, key2 = keys[i], keys[j]
            rest_keys = [x for x in keys]
            for get_rest in [key1, key2]:
                rest_keys.remove(get_rest)
            # main function: avoid a point in two points by calculating the angle
            for random in rest_keys:
                angle = calculate_angle(coordinates, random, key1, key2)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if angle > 120:  # if the program do not work well, we can adjust the value '150'
                    break
                elif random == rest_keys[-1]:
                    lines.append([key1, key2])
                    if TEST: draw_line(coordinates, key1, key2, img)
                else:
                    continue
    return lines


def slope_and_length(point1, point2):
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    if x1 != x2:
        slope = (y2-y1)/(x2-x1)
        sl_radian = np.arctan(slope)
        sl_degree = np.degrees(sl_radian)
    else:
        sl_degree = 90
    length = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return sl_degree, length


def get_skel_image(img):
    resize_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    sharp = cv2.addWeighted(gray, 1, laplacian, -0.5, 0)
    _, binary = cv2.threshold(sharp, 200, 255, cv2.THRESH_BINARY_INV)
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    skel_img = dist_on_skel.astype(np.uint8) * 255
    skel_img = cv2.resize(skel_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return skel_img


def delete_wrong_lines(lines, output_dir, img_path, coordinates, expand_factor, TEST):
    lines_after_deleting = []
    number = 0  # use to count the process of deleting wrong lines process
    for ln in lines:
        number += 1
        if number % 100 == 0: print(f'The whole process: {number}/{len(lines)}')
        p1, p2 = coordinates[ln[0]], coordinates[ln[1]]

        '''slope and line_length calculation based on 2 points'''
        sl_degree_of_2points, length_of_line = slope_and_length(p1, p2)

        '''crop a sub image and prepare for the next step'''
        x, y = min(p1[0], p2[0]), min(p1[1], p2[1])
        w, h = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
        h = 20 if h <= 20 else h  # avoid the line is a horizontal line
        w = 20 if w <= 20 else w  # avoid the line is a vertical line
        # expand_factor is defined by myself, it is a global variable
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x_direct_expand = round(w * expand_factor)
        y_direct_expand = round(h * expand_factor)
        # below: avoid the new x, y is less than zero (otherwise, it will be a bug)
        x = x - x_direct_expand if x - x_direct_expand > 0 else x
        y = y - y_direct_expand if y - y_direct_expand > 0 else y
        # problem above will not occur below, therefore, set no restriction
        w = w + x_direct_expand*2
        h = h + y_direct_expand*2
        img = cv2.imread(img_path)
        crop_img = img[y:y+h, x:x+w]

        # get skeleton of the sub image
        skel_img = get_skel_image(crop_img)
        if TEST: cv2.imwrite(f'./{output_dir}/20230301SlopeLines/{ln[0]}_{ln[1]}_1_L_0_角度{round(sl_degree_of_2points, 2)}_长度{round(length_of_line, 2)}------------------------------------↓↓↓↓↓↓.png',skel_img)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        hough_lines = cv2.HoughLinesP(image=skel_img, rho=1, theta=np.pi / 180, threshold=round(length_of_line*0.4), lines=np.array([]),
                                minLineLength=round(length_of_line*0.7), maxLineGap=round(length_of_line*0.4))
        h_l_num = 0
        try:  # use this to avoid condition that there is no line in 'hough_lines'
            for h_l in hough_lines:
                h_l_num += 1
                x1, y1, x2, y2 = h_l[0]
                sl_degree, length = slope_and_length([x1, y1], [x2, y2])
                # requirement1: represents 'sl_degree' and 'sl_degree_of_2points' are both positive(++) or negative(--)
                requirement1 = sl_degree * sl_degree_of_2points >= 0
                # requirement2: when degree close to 90 or -90, HoughLinesP may wrong (it is '+',but detect result is '-'), here to avoid it.
                requirement2 = abs(90-abs(sl_degree)) > 10
                if requirement2:  # when degree is not close to 90, requirement2 is 'True'
                    if requirement1:  # when both are ++ or --
                        difference = abs(sl_degree - sl_degree_of_2points)
                    else:
                        continue
                else:  # when degree is close to 90
                    difference = abs(abs(sl_degree_of_2points) - abs(sl_degree))
                if TEST:
                    ln_img = crop_img.copy()
                    cv2.line(ln_img, tuple([x1, y1]), tuple([x2, y2]), (0, 0, 255), 1)
                    cv2.imwrite(f'./{output_dir}/20230301SlopeLines/{ln[0]}_{ln[1]}_2_L_{h_l_num}_角度{round(sl_degree, 2)}_长度{round(length, 2)}_角差_{round(difference,2)}_长比{round(length/length_of_line,2)}.png', ln_img)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                threshold_for_def = 10
                if difference < threshold_for_def:
                    if ln in lines_after_deleting:
                        continue
                    else:
                        lines_after_deleting.append(ln)
                        if TEST: cv2.imwrite(f'./{output_dir}/20230301SlopeLines/{ln[0]}_{ln[1]}_3_L_{h_l_num}_角度{round(sl_degree, 2)}_长度{round(length, 2)}_角差_{round(difference,2)}_长比{round(length/length_of_line,2)}---------------↑↑↑↑↑↑↑success.png', ln_img)
        except:
            continue
    return lines_after_deleting


def nodes_relationship(surface_name, output_dir):

    img_path = f'./{output_dir}/{surface_name}_crop.png'

    # coordinate data
    with open(f'./{output_dir}/coordinates.json', 'r') as f:
        result = json.load(f)
    coordinates = {int(k): v for k, v in result.items()}

    adjacency_relation_with_wrong_dupl = []
    for i in range(nh):
        for j in range(nw):
            if TEST:
                i, j = count_h, count_w
            # get a sub image and its size
            sub_img, sub_size = get_ij_sub_img(img_path, nh, nw, i, j, expand_factor)  # from another py file
            if TEST: sub_img.save(f'./{output_dir}/sub_img_{i}{j}.png')
            # coordinates here represent grid nodes
            coordinates_in_sub = coordinates_in_sub_img(sub_size, coordinates)
            # lines store line, and here line represents two points(key of dictionary) that make up a line
            lines = avoid_point_in_two_points(coordinates_in_sub, img_path, surface_name, output_dir)
            print(f'The real lines(between 2 points) in sub_img_{i}{j} are:\n{lines}')
            adjacency_relation_with_wrong_dupl += lines
            if TEST: break
        if TEST: break

    # to avoid duplicated data. step 1: make sure x[0] < x[1]
    adjacency_relation_with_wrong_dupl = [[x[0], x[1]] if x[0] < x[1] else [x[1], x[0]] for x in adjacency_relation_with_wrong_dupl]
    # to avoid duplicated data. step 2: delete  the duplicated
    adjacency_relation_with_wrong = []
    for i in adjacency_relation_with_wrong_dupl:
        if i in adjacency_relation_with_wrong:
            continue
        else:
            adjacency_relation_with_wrong.append(i)
    adjacency_relation = delete_wrong_lines(adjacency_relation_with_wrong, output_dir, img_path, coordinates, expand_factor, TEST)
    with open(f'./{output_dir}/adjacency_relation.json', 'w') as fp:
        json.dump(adjacency_relation, fp, indent=4)
    print('The final result is:\n ', adjacency_relation)
    print(f'The number of lines before deleting the wrong lines: {len(adjacency_relation_with_wrong)}')
    print('Total number of lines in grid is: ', len(adjacency_relation))


'''TEST MODE or NOT'''
TEST = 0  # '1' is test mode, '0' is ordinary mode
TEST = True if TEST == 1 else False
count_h, count_w = 3, 6  # define which area to test: count_h < nh, count_w < w

# the number in two direction and expand_factor
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
nh: int = 4
nw: int = 8
expand_factor: float = 0.3  # 0%

if __name__ == '__main__':
    nodes_relationship('D23', './Surface_D23')
