import N1_CropToLeftGrid
import N2_CornerHarrisPoints
import N4_NodesRelationship
import N5_ResultVisualization
import N6_UseCoordTopoDrawGrid

surface_name = '4-D05'

output_dir = 'Surface' + '_' + surface_name

# the rectangle boundary is useless, therefore we need to crop the image and only left the grid
N1_CropToLeftGrid.crop(surface_name, output_dir)

# extract the nodes of the grid, namely extract pixel coordinates
N2_CornerHarrisPoints.corner_harris_points(surface_name, output_dir, test_or_not=False)

# extract the nodes relationship
N4_NodesRelationship.nodes_relationship(surface_name, output_dir)

# here is to visualize the result of extracted nodes relationship
# this part is not necessary to get the final result(it can be skipped)
N5_ResultVisualization.result_visualize(surface_name, output_dir)

# use coordinates and topology relationship to get the grid(planar grid)
N6_UseCoordTopoDrawGrid.coord_topo_2_grid(output_dir)

