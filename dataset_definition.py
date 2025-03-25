from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models.detector.icecube import IceTop
from graphnet.models.graphs  import  KNNGraph
from graphnet.models.graphs.nodes  import  NodesAsPulses

graph_definition = KNNGraph(
    detector=IceTop(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
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
        "train": "100 random events ~ event_no % 2 == 0",
        "test": "event_no % 2 == 1",
}

graph = dataset[0]  # torch_geometric.data.Data
dataset.config.dump("test.yml")
