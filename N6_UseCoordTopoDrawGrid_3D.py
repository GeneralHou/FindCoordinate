import json
import plotly.graph_objs as go

def draw_grid_3d_space(surface_name):
    output_dir = 'Surface' + '_' + surface_name
    with open(f'./{output_dir}/adjacency_relation.json', 'r') as f:
        adjacency_relation = json.load(f)

    with open(f'./{output_dir}/coordinates_space.json', 'r') as f:
        coordinates_space = json.load(f)

    # Create plot points
    scatter_points = []
    for key, value in coordinates_space.items():
        scatter_points.append(
            go.Scatter3d(
                x=[value[0]],
                y=[value[1]],
                z=[value[2]],
                mode='markers',
                marker=dict(
                    size=3,
                    color='red'
                ),
                text=str(key),
                name='Point ' + str(key)
            )
        )

    # Create a list of line plot edges
    line_edges = []
    for edge in adjacency_relation:
        line_edges.append(
            go.Scatter3d(
                x=[coordinates_space[str(edge[0])][0], coordinates_space[str(edge[1])][0]],
                y=[coordinates_space[str(edge[0])][1], coordinates_space[str(edge[1])][1]],
                z=[coordinates_space[str(edge[0])][2], coordinates_space[str(edge[1])][2]],
                mode='lines',
                line=dict(
                    width=2,
                    color='blue'
                ),
               showlegend=False
            )
        )

    # Create the layout for the plot
    layout = go.Layout(
        title=f'Visualized result of {surface_name}',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        showlegend=False
    )

    # Create the figure and add the scatter points and line edges
    fig = go.Figure(data=scatter_points + line_edges, layout=layout)

    # Save the figure as an HTML file
    fig.write_html(f'./{output_dir}/Visualized_{surface_name}.html')


if __name__ == '__main__':
    draw_grid_3d_space(surface_name='4-000')
