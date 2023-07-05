import networkx as nx
import matplotlib.pyplot as plt
import json


class CoordTopo2grid:
    def __init__(self, output_dir, final_img_name='N5_FinalResult', show_text=False):
        self.output_dir = output_dir
        self.final_img_name = final_img_name
        self.show_text = show_text

    def run(self):
        with open(f'{self.output_dir}/coordinates.json', 'r') as f:
            coordinates = json.load(f)
        coordinates = {int(k): v for k, v in coordinates.items()}

        with open(f'{self.output_dir}/adjacency_relation.json', 'r') as f:
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

        # add text annotations
        if self.show_text:
            for k, v in coordinates.items():
                x, y = v[0], -v[1]
                plt.text(x, y, str(k), fontsize=4, ha='left', va='center', color='blue')

        plt.savefig(f'{self.output_dir}/{self.final_img_name}.jpg', dpi=1200)


if __name__ == '__main__':
    Grid2d = CoordTopo2grid(output_dir='./Surface_4-000', final_img_name='N5_FinalResult')
    Grid2d.run()