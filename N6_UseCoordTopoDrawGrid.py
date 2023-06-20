import networkx as nx
import matplotlib.pyplot as plt
import json


def coord_topo_2_grid(output_dir):
    with open(f'{output_dir}/coordinates.json', 'r') as f:
        coordinates = json.load(f)
    coordinates = {int(k): v for k, v in coordinates.items()}

    with open(f'{output_dir}/adjacency_relation.json', 'r') as f:
        adjacency_relation = json.load(f)

    G = nx.Graph()

    # allocate node coordinates
    for k, v in coordinates.items():
        # we make a little change in y-axis direction for opencv and networkx use different way to show y-axis
        x, y = v[0], -v[1]
        G.add_node(k, pos=(x, y))

    # allocate  adjacency relationship (Topology relationship)
    G.add_edges_from(adjacency_relation)


    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, node_color='red', node_size=1, edge_color='black', width=0.5)
    plt.savefig(f'{output_dir}/FinalResult.png', dpi=1200)


if __name__ == '__main__':
    coord_topo_2_grid('./Surface_D23')