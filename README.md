# FindCoordinate
pycharmforGHfirstProject\20230102FindCoordinate

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

------------------------------------------
■■■■20230629：fixbug: the code in this main branch can now work well with S19_0 grid. Meanwhile, I also think it is because threshold in N2*.py plays an important role in determining the nodes find ability.