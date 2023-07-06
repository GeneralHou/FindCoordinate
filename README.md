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