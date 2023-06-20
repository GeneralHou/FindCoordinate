import cv2
import datetime
import json


def result_visualize(surface_name, output_dir):
    # coordinate data
    with open(f'./{output_dir}/coordinates.json', 'r') as f:
        result = json.load(f)
    coordinates = {int(k): v for k, v in result.items()}

    # read adjacency_relation from 'json' file
    with open(f'./{output_dir}/adjacency_relation.json', 'r') as f:
        adjacency_relation = json.load(f)

    img = cv2.imread(f'./{output_dir}/{surface_name}_crop.png', 1)

    for rlt in adjacency_relation:
        pt1 = tuple(int(x) for x in coordinates[rlt[0]])
        pt2 = tuple(int(x) for x in coordinates[rlt[1]])
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)

        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # cv2.imshow('Image', img)
        # cv2.waitKey(750)
        # cv2.destroyAllWindows()

    time = datetime.datetime.now().strftime("%m%d%H%M_")
    cv2.imwrite(f'./{output_dir}/{time}Visualization_{len(adjacency_relation)}lines.png', img)


if __name__ == '__main__':
    result_visualize('D23', 'Surface_D23')
