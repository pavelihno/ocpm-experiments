import os
import webbrowser

import gravis as gv
import networkx as nx
import pandas as pd
import pm4py
from networkx.algorithms import bipartite
from pyvis.network import Network


def load_ocel(ocel_path):
    """
    Load an Object-Centric Event Log (OCEL) from a given file path.

    Parameters:
    - ocel_path (str): The file path to the OCEL data.

    Returns:
    - ocel (OCEL): An instance of the OCEL class containing the loaded data.
    """
    ocel = pm4py.read_ocel2(ocel_path)
    return ocel


def create_ocel_graph(ocel_path):
    """
    Build a heterogeneous NetworkX graph from an OCEL.

    Parameters:
    - ocel_path (str): The file path to the OCEL data.

    Returns:
    - G (networkx.Graph): A graph containing both event and object nodes, with
      edges added for every event-object relation.
    - objects_df (pd.DataFrame): DataFrame with object attributes from the OCEL.
    - events_df (pd.DataFrame): DataFrame with event attributes from the OCEL.
    - relations_df (pd.DataFrame): DataFrame with event-object relationships.
    """
    ocel = load_ocel(ocel_path)

    events_df = ocel.events
    objects_df = ocel.objects
    relations_df = ocel.relations

    # Create a heterogeneous graph
    G = nx.Graph()

    # Add event nodes
    for _, row in events_df.iterrows():
        attributes = {k: str(v) for k, v in row.dropna().to_dict().items()}
        attributes['node_type'] = 'event'

        G.add_node(row['ocel:eid'], **attributes)

    # Add object nodes
    for _, row in objects_df.iterrows():
        attributes = {k: str(v) for k, v in row.dropna().to_dict().items()}
        attributes['node_type'] = 'object'

        G.add_node(row['ocel:oid'], **attributes)

    # Add edges representing the relations (Event <-> Object)
    for _, row in relations_df.iterrows():
        G.add_edge(row['ocel:eid'], row['ocel:oid'])

    return G, objects_df, events_df, relations_df


def create_object_graph(G, allowed_types=None, weighted=False):
    """
    Project a bipartite event-object graph onto a homogeneous graph.

    Parameters:
    - G (networkx.Graph): The heterogeneous graph containing both event and
      object nodes.
    - allowed_types (list, optional): A list of allowed object types to include in the
      projected graph. If None, all object types are included.
    - weighted (bool): When True, keeps edge weights based on the number of
      shared neighbors.

    Returns:
    - projected_graph (networkx.Graph): A homogeneous projected graph restricted
      to nodes of the requested type.
    """
    allowed_types = allowed_types or []

    target_nodes = [
        n
        for n, attr in G.nodes(data=True)
        if attr.get('node_type') == 'object'
        and (not allowed_types or attr.get('ocel:type') in allowed_types)
    ]

    event_nodes = [
        n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'event'
    ]

    filtered_G = G.subgraph(target_nodes + event_nodes)

    if weighted:
        projected_graph = bipartite.weighted_projected_graph(filtered_G, target_nodes)
    else:
        projected_graph = bipartite.projected_graph(filtered_G, target_nodes)

    return projected_graph


def create_event_graph(
    G, nodes=None, weighted=False, include_objects=False, allowed_obj_types=None
):
    """Build a directed event-to-event graph from a heterogeneous graph.

    Args:
        G (networkx.Graph): Heterogeneous graph containing event and object
            nodes.
        nodes (iterable, optional): Restrict the construction to a subset of
            nodes. If `None`, all nodes are considered.
        weighted (bool): If `True`, add a `weight` attribute to each edge equal
            to the number of shared objects linking the consecutive events.
        include_objects (bool): If `True`, keeps the object nodes in the
            resulting graph and adds edges from events to their related objects.
        allowed_obj_types (list, optional): A list of allowed object types
            to consider when building event connections or
            including objects. If None, all object types are included.

    Returns:
        networkx.DiGraph: Directed graph whose nodes are events (and optionally
        objects). An edge from one event to the next indicates that both events
        share an allowed object and are ordered by timestamp.
    """
    if nodes is None:
        target_nodes = G.nodes()
    else:
        target_nodes = nodes

    allowed_obj_types = allowed_obj_types or []

    event_nodes = [n for n in target_nodes if G.nodes[n].get('node_type') == 'event']

    DiG = nx.DiGraph()
    for event_id in event_nodes:
        DiG.add_node(event_id, **G.nodes[event_id])

    for node_id, attr in G.nodes(data=True):
        if attr.get('node_type') == 'object':
            if allowed_obj_types and attr.get('ocel:type') not in allowed_obj_types:
                continue

            connected_events = [n for n in G.neighbors(node_id) if n in event_nodes]

            if include_objects and connected_events:
                DiG.add_node(node_id, **attr)
                for ev in connected_events:
                    # Edges in both directions
                    DiG.add_edge(ev, node_id, edge_type='event_to_object')
                    DiG.add_edge(node_id, ev, edge_type='event_to_object')

            if len(connected_events) > 1:
                # Events sorted chronologically and event ID as a tiebreaker
                connected_events.sort(
                    key=lambda x: (pd.to_datetime(G.nodes[x]['ocel:timestamp']), x)
                )

                for i in range(len(connected_events) - 1):
                    source_event = connected_events[i]
                    target_event = connected_events[i + 1]

                    if DiG.has_edge(source_event, target_event):
                        if 'shared_objects' in DiG[source_event][target_event]:
                            DiG[source_event][target_event]['shared_objects'].append(
                                node_id
                            )
                        if weighted and 'weight' in DiG[source_event][target_event]:
                            DiG[source_event][target_event]['weight'] += 1
                    else:
                        edge_attrs = {
                            'shared_objects': [node_id],
                            'edge_type': 'directly_follows',
                        }
                        if weighted:
                            edge_attrs['weight'] = 1
                        DiG.add_edge(source_event, target_event, **edge_attrs)

    return DiG


def display_pyvis(
    graph,
    event_color='#90caf9',
    obj_color='#a5d6a7',
    highlight_color='#ff7f7f',
    obj_type_to_color=None,
    highlight_node_id=None,
    output_file_path='outputs/graph_display.html',
    height='800px',
    width='100%',
    show_buttons=True,
):
    """
    Render a NetworkX graph in a PyVis HTML view.

    Parameters:
    - graph (networkx.Graph or networkx.DiGraph): The graph to visualize. Nodes are
    expected to include a 'node_type' attribute, and object nodes may also have an
      'obj_class' or 'ocel:type' attribute.
    - directed (bool): If True, renders the graph with directed edges (arrows).
    - event_color (str): Fill color for event nodes.
    - obj_color (str): Default fill color for object nodes.
    - highlight_color (str): Fill color for a highlighted node.
    - obj_type_to_color (dict, optional): Mapping from object type to custom
      colors; used for object nodes when available.
    - highlight_node_id (hashable, optional): Node ID to highlight.
    - output_file_path (str): Destination HTML path for the saved visualization.
    - height (str): Height of the rendered PyVis canvas.
    - width (str): Width of the rendered PyVis canvas.
    - show_buttons (bool): If True, shows the PyVis control buttons.

    Returns:
    - None. The graph is saved to an HTML file and opened in the default browser.
    """
    if obj_type_to_color is None:
        obj_type_to_color = {}

    G = graph.copy()

    edges_to_remove = []
    seen_undirected = set()

    for u, v, edge_data in G.edges(data=True):
        if edge_data.get('edge_type') == 'event_to_object':
            if (v, u) in seen_undirected:
                edges_to_remove.append((u, v))
            else:
                seen_undirected.add((u, v))

    G.remove_edges_from(edges_to_remove)

    net = Network(
        height=height,
        width=width,
        bgcolor='#ffffff',
        font_color='black',
        notebook=False,
        cdn_resources='in_line',
        directed=isinstance(graph, nx.DiGraph),
    )

    # Node styles
    for node in G.nodes():
        node_info = G.nodes[node]

        if node_info.get('node_type') == 'event':
            color = event_color
        elif node == highlight_node_id:
            color = highlight_color
        else:
            obj_type = node_info.get('obj_class') or node_info.get('ocel:type')
            color = obj_type_to_color.get(obj_type, obj_color)

        G.nodes[node]['color'] = color

        G.nodes[node]['label'] = (
            node_info.get('ocel:activity')
            if node_info.get('node_type') == 'event'
            else str(node)
        )

        hover_text = f'ID: {node}\n'
        hover_text += '—' * 20 + '\n'
        for key, value in node_info.items():
            if key not in ['color', 'label', 'size', 'title', 'hover', 'node_type']:
                hover_text += f'{key}: {value}\n'

        G.nodes[node]['title'] = hover_text

    # Edge styles
    for u, v, edge_data in G.edges(data=True):
        edge_type = edge_data.get('edge_type')

        if edge_type == 'event_to_object':
            edge_data['arrows'] = ''
            edge_data['dashes'] = True

        elif edge_type == 'directly_follows':
            edge_data['arrows'] = 'to'

            if 'weight' in edge_data:
                edge_data['label'] = str(edge_data['weight'])

    if show_buttons:
        net.show_buttons(filter_=['physics'])

    net.from_nx(G)

    html_content = net.generate_html()

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    webbrowser.open(f'file://{os.path.abspath(output_file_path)}')


def display_gravis(
    graph,
    highlight_node_id=None,
    obj_type_to_color=None,
    event_color='#90caf9',
    obj_color='#a5d6a7',
    highlight_color='#ff7f7f',
    highlight_size=20,
    graph_height=800,
):
    """
    Display a NetworkX graph in a 3D Gravis view.

    Parameters:
    - graph (networkx.Graph or networkx.DiGraph): The graph to visualize. Nodes should
      have a 'node_type' attribute, and object nodes may also have an 'obj_class'
      or 'ocel:type' attribute.
    - highlight_node_id (hashable, optional): Node ID to highlight visually.
    - obj_type_to_color (dict, optional): Mapping from object type to custom colors.
    - event_color (str): Fill color for event nodes.
    - obj_color (str): Default fill color for object nodes.
    - highlight_color (str): Fill color for the highlighted node.
    - highlight_size (int): Size to assign to the highlighted node.
    - graph_height (int): Height of the Gravis visualization area.
    - directed (bool): If True, render edges as directed links.

    Returns:
    - None. The graph is displayed interactively via Gravis.
    """
    if obj_type_to_color is None:
        obj_type_to_color = {}

    G = graph.copy()

    for node in G.nodes():
        node_info = G.nodes[node]

        # Color nodes
        if node_info.get('node_type') == 'event':
            G.nodes[node]['color'] = event_color
        elif node == highlight_node_id:
            G.nodes[node]['color'] = highlight_color
            G.nodes[node]['size'] = highlight_size
        else:
            obj_type = node_info.get('obj_class') or node_info.get('ocel:type')
            G.nodes[node]['color'] = obj_type_to_color.get(obj_type, obj_color)

        # Label nodes
        G.nodes[node]['label'] = (
            node_info.get('ocel:activity')
            if node_info.get('node_type') == 'event'
            else str(node)
        )

        # Hover text
        hover_html = "<div style='font-family: Arial; font-size: 12px;'>"
        hover_html += f'<b>ID:</b> {node}<br>'
        hover_html += "<hr style='margin: 4px 0;'>"

        for key, value in node_info.items():
            if key not in ['color', 'label', 'size', 'title', 'hover', 'node_type']:
                hover_html += f'<b>{key}:</b> {value}<br>'

        hover_html += '</div>'
        G.nodes[node]['hover'] = hover_html

    fig = gv.three(
        G,
        graph_height=graph_height,
        node_label_data_source='label',
    )
    fig.display()
