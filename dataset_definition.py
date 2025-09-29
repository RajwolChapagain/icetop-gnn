import argparse
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models.detector.icecube import IceTop
from graphnet.models.graphs  import  KNNGraph
from graphnet.models.graphs.nodes  import  NodesAsPulses

parser = argparse.ArgumentParser(description="Dump GraphNeT dataset config")
parser.add_argument(
    "--output", "-o",
    type=str,
    required=True,
    help="Filename under configs folder to save the dataset config YAML"
)
args = parser.parse_args()

graph_definition = KNNGraph(
    detector=IceTop(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=32,
)

dataset = SQLiteDataset(
    path="data_out/merged/merged.db",
    graph_definition=graph_definition,
    pulsemaps="OfflineIceTopHLCTankPulses",
    features=["charge", "dom_time", "dom_x", "dom_y"], 
    truth_table="truth_table",
    truth=["energy"],
)

dataset.config.selection = {
        "lean": "30000 random events ~",
}

dataset.config.dump(f"configs/{args.output}")
print(f"Dataset config dumped to configs/{args.output}")