import N1_CropToLeftGrid
import N2_ExtractCoordinate
import N3_NodesRelationship
import N4_RemoveAddRelation
import N5_UseCoordTopoDrawGrid

surface_name = 'S19_0'

output_dir = 'Surface' + '_' + surface_name

# the rectangle boundary is useless, therefore we need to crop the image and only left the grid
N1_CropToLeftGrid.crop(surface_name)

# extract the nodes of the grid, namely extract pixel coordinates
N2_ExtractCoordinate.extract_coordinates(surface_name)

# extract the nodes relationship
N3_NodesRelationship.nodes_relationship(surface_name)

# fix the uncorrect nodes(coordinates) relationship
N4_RemoveAddRelation.Rmv_add_relatn(surface_name, final_img_name="N4_GridWithKeys")

# use coordinates and topology relationship to get the grid(planar grid)
N5_UseCoordTopoDrawGrid.CoordTopo2grid(output_dir, show_text=False).run()

# Prompt:
print("If there still exists any wrong relationship, just run 'N4_RemoveAddRelation.py' seperately.")