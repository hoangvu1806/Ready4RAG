"""
Visualize the knowledge graph from a GraphML file.

Usage:
    python -m rag_ingest.visualize_graph [graphml_path] [output_image]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

# Ensure a font that supports the document's character set is selected.
plt.rcParams["font.family"] = "sans-serif"

# Entity type to node color mapping.
_COLOR_MAP = {
    "PERSON": "skyblue",
    "LOCATION": "lightgreen",
    "CONCEPT": "orange",
    "TECHNOLOGY": "mediumpurple",
    "ORGANIZATION": "salmon",
    "EVENT": "gold",
}
_DEFAULT_COLOR = "lightgray"


def visualize_graph(
    graphml_path: str,
    output_image: str = "knowledge_graph.png",
) -> None:
    """
    Render the knowledge graph to a PNG image.

    Args:
        graphml_path: Path to the GraphML file.
        output_image: Destination image path.
    """
    src = Path(graphml_path)
    if not src.exists():
        raise FileNotFoundError(f"GraphML file not found: {src}")

    print(f"Loading graph from {src} ...")
    graph = nx.read_graphml(str(src))

    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    if graph.number_of_nodes() == 0:
        print("Graph is empty, nothing to render.")
        return

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(graph, k=0.6, iterations=60, seed=42)

    node_colors = [
        _COLOR_MAP.get(data.get("entity_type", "").upper(), _DEFAULT_COLOR)
        for _, data in graph.nodes(data=True)
    ]

    nx.draw_networkx_nodes(
        graph, pos, node_size=1200, node_color=node_colors, alpha=0.85
    )
    nx.draw_networkx_edges(
        graph, pos, width=0.8, alpha=0.4, edge_color="gray", arrows=True
    )
    labels = {
        node: data.get("entity_name", node)
        for node, data in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(graph, pos, labels, font_size=7, font_weight="bold")

    edge_labels = {
        (u, v): data.get("relationship_type", "")
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels, font_size=6, alpha=0.7
    )

    plt.title("Knowledge Graph", fontsize=14)
    plt.axis("off")

    plt.savefig(output_image, format="png", dpi=200, bbox_inches="tight")
    print(f"Graph image saved: {output_image}")


def main() -> None:
    graphml = sys.argv[1] if len(sys.argv) > 1 else "database/graph/knowledge_graph.graphml"
    output = sys.argv[2] if len(sys.argv) > 2 else "knowledge_graph.png"
    visualize_graph(graphml, output)


if __name__ == "__main__":
    main()
