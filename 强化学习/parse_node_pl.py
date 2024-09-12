class Node:
    def __init__(self, node_name, width, height, movetype=None, ll_xcoord=None, ll_ycoord=None):
        self.node_name = node_name
        self.width = width
        self.height = height
        self.movetype = movetype if movetype else "movable"
        self.ll_xcoord = ll_xcoord
        self.ll_ycoord = ll_ycoord


class Nodes:
    def __init__(self):
        self.numnodes = 0
        self.numterminals = 0
        self.node_list = []
        self.coords_max=[0,0,0,0]   #四个数值依次为：所有节点的x的最大值，y的最大值，宽度的最大值，高度的最大值，用于确定二维网格表的大小

def parse_nodes_file(nodes_file):
    nodes_data = {}
    num_nodes = 0
    num_terminals = 0
    ignore_line = False

    with open(nodes_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if line.startswith("NumNodes"):
                num_nodes = int(line.split(":")[1].strip())
            elif line.startswith("NumTerminals"):
                num_terminals = int(line.split(":")[1].strip())
                break

        node_count = 0
        terminal_count = 0
        for line in lines:
            if line.startswith("#") or not line.strip() or line.startswith("NumNodes") or line.startswith("NumTerminals") or ignore_line:
                ignore_line = False
                continue

            items = line.strip().split()
            if len(items) < 3:
                continue

            # Ignore the line that needs to be skipped
            if items[0] == "UCLA" and items[1] == "nodes" and items[2] == "1.0":
                ignore_line = True
                continue

            node_name = items[0]
            width = items[1]
            height = items[2]
            node_type = "terminal" if len(items) > 3 and items[3] == "terminal" else "movable"

            nodes_data[node_name] = (width, height, node_type)

            if node_type == "terminal":
                terminal_count += 1

            node_count += 1

            if node_count >= num_nodes:
                break

        return nodes_data

def parse_pl_file(pl_file):
    pl_data = {}
    with open(pl_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            items = line.strip().split()
            if len(items) >= 4:
                node_name, ll_xcoord, ll_ycoord = items[:3]
                pl_data[node_name] = (ll_xcoord, ll_ycoord)
    return pl_data


def fill_nodes_structure(nodes, nodes_file, pl_file):
    nodes_data = parse_nodes_file(nodes_file)
    pl_data = parse_pl_file(pl_file)

    for node_name, node_info in nodes_data.items():
        width, height, movetype = node_info
        ll_xcoord, ll_ycoord = pl_data.get(node_name, (None, None))
        new_node = Node(node_name, int(width), int(height), movetype, int(ll_xcoord), int(ll_ycoord))
        nodes.node_list.append(new_node)

    nodes.numnodes = len(nodes_data)
    nodes.numterminals = sum(1 for node in nodes_data.values() if node[2] in ["terminal", "moveable", "fixed"])  # Counting all types of nodes


    return nodes


# Example usage
nodes = Nodes()
nodes_file = "test.nodes"
pl_file = "test.pl"
nodes = fill_nodes_structure(nodes, nodes_file, pl_file)

'''
for i in nodes.node_list:
    print(i.node_name)


if nodes.node_list:
    first_node = nodes.node_list[0]
    print(vars(first_node))
if nodes.node_list:
    for node in nodes.node_list:
        print(vars(node))
print(f"Number of nodes: {nodes.numnodes}")
print(f"Number of terminals: {nodes.numterminals}")
'''