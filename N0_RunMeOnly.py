import N1_CropToLeftGrid
import N2_ExtractCoordinate
import N3_NodesRelationship
import N4_RemoveAddRelation
import N5_UseCoordTopoDrawGrid

surface_name = '1-000'

output_dir = 'Surface' + '_' + surface_name

# the rectangle boundary is useless, therefore we need to crop the image and only left the grid
print('■'*5, 'STEP 1: CROP IMAGE')
N1_CropToLeftGrid.crop(surface_name)

print()
print('■'*5, 'STEP 2: EXTRACT COORDINATES')
# extract the nodes of the grid, namely extract pixel coordinates
N2_ExtractCoordinate.extract_coordinates(surface_name)

print()
print('■'*5, 'STEP 3: EXTRACT RELATIONSHIP')
# extract the nodes relationship
N3_NodesRelationship.nodes_relationship(surface_name)

print()
print('■'*5, 'STEP 4: FIX THE BUG MAY EXIST')
# fix the incorrect nodes(coordinates) relationship
N4_RemoveAddRelation.Rmv_add_relatn(surface_name, final_img_name="N4_GridWithKeys")

print()
print('■'*5, 'STEP 5: DRAW THE FINAL RESULT')
# use coordinates and topology relationship to get the grid(planar grid)
N5_UseCoordTopoDrawGrid.CoordTopo2grid(output_dir, show_text=False).run()

# Prompt:
print("If there still exists any wrong relationship, just run 'N4_RemoveAddRelation.py' separately.")