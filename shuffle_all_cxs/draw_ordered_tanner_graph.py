import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from simultaneous_scheduling import schedule_gates


def draw_cx_list(cx_list, ancilla_type, data_coords=None, ancilla_coords=None):
    """
    Draw the cx_list as a tripartite graph.

    Nodes are divided into three columns by default:
      - X ancilla nodes at x = 0.
      - Data qubits at x = 1.
      - Z ancilla nodes at x = 2.

    Each edge represents all CX gates between a data qubit and an ancilla.
    The edge label is a comma-separated list of the order indices from cx_list.

    Optionally, custom coordinates for data qubits and ancillas can be provided.

    Parameters:
      cx_list : list of tuple
          Each tuple (q, a) represents a CX gate, where q is the data qubit and a is the ancilla.
      ancilla_type : dict
          Dictionary mapping each ancilla id to its type ("X" or "Z").
      data_coords : dict, optional
          Dictionary mapping each data qubit id to its (x, y) coordinate.
      ancilla_coords : dict, optional
          Dictionary mapping each ancilla id to its (x, y) coordinate.
    """
    # Collapse the cx_list into a dictionary:
    # # key: (data, ancilla), value: list of order indices (as strings)
    # edge_dict = {}
    # for idx, (q, a) in enumerate(cx_list):
    #     key = (q, a)
    #     edge_dict.setdefault(key, []).append(str(idx))
    # # Build labels for each edge.
    # edge_labels = {key: ", ".join(labels) for key, labels in edge_dict.items()}

    edge_labels = schedule_gates(cx_list)
    edge_labels = {k: str(v) for k, v in edge_labels.items()}


    # Create a simple graph and add nodes.
    G = nx.Graph()

    # Collect nodes from cx_list.
    data_nodes = {q for q, _ in cx_list}
    ancilla_nodes = {a for _, a in cx_list}
    # Partition ancilla nodes by type.
    x_ancillas = {a for a in ancilla_nodes if ancilla_type.get(a) == "X"}
    z_ancillas = {a for a in ancilla_nodes if ancilla_type.get(a) == "Z"}

    # Add nodes with group attributes.
    for d in data_nodes:
        G.add_node(d, group="data")
    for a in x_ancillas:
        G.add_node(a, group="x_ancilla")
    for a in z_ancillas:
        G.add_node(a, group="z_ancilla")

    # Add one edge per (data, ancilla) pair.
    for (q, a), label in edge_labels.items():
        G.add_edge(q, a, label=label)

    # If custom coordinates are not provided, compute default positions.
    pos = {}
    if data_coords is None or ancilla_coords is None:
        # Set default x positions for each group.
        x_positions = {"x_ancilla": 0, "data": 1, "z_ancilla": 2}

        def assign_positions(nodes, x_val):
            nodes = sorted(nodes)
            n = len(nodes)
            # Evenly space y coordinates from 1 to -1.
            y_vals = np.linspace(1, -1, n)
            return {node: (x_val, y) for node, y in zip(nodes, y_vals)}

        default_data_coords = assign_positions(data_nodes, x_positions["data"])
        default_x_coords = assign_positions(x_ancillas, x_positions["x_ancilla"])
        default_z_coords = assign_positions(z_ancillas, x_positions["z_ancilla"])

        # Use custom coordinates if provided; otherwise use defaults.
        if data_coords is None:
            data_coords = default_data_coords
        if ancilla_coords is None:
            # Merge X and Z ancilla defaults.
            ancilla_coords = {**default_x_coords, **default_z_coords}

    # Combine provided coordinates into pos.
    pos.update(data_coords)
    pos.update(ancilla_coords)

    # Define node colors based on group.
    color_map = {
        "data": "lightblue",
        "x_ancilla": "lightgreen",
        "z_ancilla": "lightcoral"
    }
    node_colors = [color_map.get(G.nodes[node]["group"], "gray") for node in G.nodes()]

    # Draw the graph.
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
    nx.draw_networkx_edges(G, pos, connectionstyle="arc3, rad=0.1")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color="red")

    plt.axis("off")
    plt.title("Tripartite Graph Representation of cx_list")
    plt.show()


# --- Example usage ---
if __name__ == "__main__":
    # Example cx_list for a 3x3 rotated surface code.
    cx_list = [
        # X stabilizer 0:
        (0, "aX0"), (1, "aX0"), (3, "aX0"), (4, "aX0"),
        # X stabilizer 1:
        (2, "aX1"), (5, "aX1"),
        # X stabilizer 2:
        (3, "aX2"), (6, "aX2"),
        # X stabilizer 3:
        (4, "aX3"), (5, "aX3"), (7, "aX3"), (8, "aX3"),
        # Z stabilizer 0:
        (0, "aZ0"), (1, "aZ0"),
        # Z stabilizer 1:
        (1, "aZ1"), (2, "aZ1"), (4, "aZ1"), (5, "aZ1"),
        # Z stabilizer 2:
        (3, "aZ2"), (4, "aZ2"), (6, "aZ2"), (7, "aZ2"),
        # Z stabilizer 3:
        (7, "aZ3"), (8, "aZ3")
    ]

    # Build the ancilla type dictionary.
    ancilla_type = {}
    for i in range(4):
        ancilla_type[f"aX{i}"] = "X"
    for i in range(4):
        ancilla_type[f"aZ{i}"] = "Z"

    # Optional: custom coordinates.
    # For example, place data qubits on a grid (here same as default) and ancillas at fixed positions.
    custom_data_coords = {
        0: (1, 1),
        1: (2, 1),
        2: (3, 1),
        3: (1, 0),
        4: (2, 0),
        5: (3, 0),
        6: (1, -1),
        7: (2, -1),
        8: (3, -1)
    }
    custom_ancilla_coords = {
        "aX0": (0, 0.5),
        "aX1": (0, -0.5),
        "aX2": (0, -1.5),
        "aX3": (0, -2.5),
        "aZ0": (4, 0.5),
        "aZ1": (4, -0.5),
        "aZ2": (4, -1.5),
        "aZ3": (4, -2.5)
    }

    # To use the default layout, simply call:
    # draw_cx_list(cx_list, ancilla_type)
    #
    # To use custom coordinates, call:
    draw_cx_list(cx_list, ancilla_type, data_coords=custom_data_coords, ancilla_coords=custom_ancilla_coords)
