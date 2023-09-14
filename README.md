# FindCoordinate
location: pycharmforGHfirstProject\20230102FindCoordinate
virtualenv: the virtual environment I used in my seu computer is: dataaug

## how to use FindCoordinates:
Step1: create a directory and name it Surface_***
       here, *** is the gan generated img without .jpg extension

Step2: put two images, the gan generated img(with red dots that specially generated, it should be named as ***_red.jpg) and the generated image without red dots(it should be named as ***.jpg), into the directory above

Step3: open N0_RunMeOnly.py and change the string pass to variable "surface_name"

Step4: clik Run

-----------------------------------------
■■■ If we want to visualize the grid in 3d:
we need to put "coordinates_space.json" under the folder "Surface_***" first
and then run the N7_UseCoordTopoDrawGrid_3D.py

-----------------------------------------
## Modification Log:
20230604: [Modify]
Total modified all the files: 
1)add gan_aid to extract nodes; 
2)add interaction part to remove and delete wrong relationship.

-----------------------------------------
20230706: [BigChange] 
1. We use a new way to find the relationship between nodes, namely search pix by pix to find the relationship.
2. At the same time, we modify other .py files to make them more suitable for the new modification.

■■ Attention: to run the new code, two images needed to be prepared(one with red dots and one without). How to use the new version is shown above.

-----------------------------------------
20230708: [BigChange] 
1. in N1_crop.py, we larger the image with 15 times in width and height direction, respectively.
2. in N3.relationship.py, we add a sub-function in "def black_lines_corresponding_centers(img, anchor)" which we call it "def erosion(img)". check the source to know more.
3. we also make some other changes, but not that big.

20230915: [Modify]
1. in N5.py, when always set the directions as 1, as below, it may face a sitution that it can not find out the next point, and loop infinitely.
[[1,1], [1,-1], [-1,-1], [-1,1],[0, 1], [0, -1], [1, 0], [-1, 0]]
2. therefore, I modfied the code, when all the direcions have been used but still can not find out the next point, enlarge the direction to 2, if still not find the next point, go to 3.