# FindCoordinate
pycharmforGHfirstProject\20230102FindCoordinate
----------------------------------------
DATE:2023.06.24:
Total modified all the files: 
1)add gan_aid to extract nodes; 
2)add interaction part to remove and delete wrong relationship.

■■ the virtual environment I used in my seu computer is: dataaug
----------------------------------------
how to use FindCoordinates:
Step1: create a directory and name it Surface_***
       here, *** is the gan generated img without .png extension

Step2: put the gan generated img into the directory above (the image nama is ***.img)

Step3: open N0_RunMeOnly.py and change the string pass to variable "surface_name"

Step4: clik Run

-----------------------------------------

■■■■ If we want to visualize the grid in 3d:
we need to put "coordinates_space.json" under the folder "Surface_***" first
and then run the N7_UseCoordTopoDrawGrid_3D.py
