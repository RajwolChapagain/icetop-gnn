from graphnet.data.extractors.icecube.ittruthextractor import ITTruthExtractor
from graphnet.data.extractors.icecube.i3featureextractor import ITFeatureExtractor
from graphnet.data.dataconverter import DataConverter
from graphnet.data.readers import I3Reader
from graphnet.data.writers import SQLiteWriter
import argparse
import os
import multiprocessing

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--indir", "-i",
    type=str,
    default="/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012/oldstructure/12360",
    help="Directory containing the i3 files (default: %(default)s)"
)

parser.add_argument(
    "--outdir", "-o",
    type=str,
    required=True,
    help="Empty directory where output .db files are saved"
)

args = parser.parse_args()
os.makedirs(args.outdir)

# Instantiate DataConverter - exports data from I3 to SQLite
converter = DataConverter(file_reader = I3Reader('/data/user/rchapagain/GeoCalibDetectorStatus_2012.56063_V1_OctSnow_scint.i3'),
                          save_method = SQLiteWriter(),
                          extractors=[ITTruthExtractor(name = "truth_table"), ITFeatureExtractor(pulsemap = "OfflineIceTopSLCTankPulses"), ITFeatureExtractor(pulsemap = "OfflineIceTopHLCTankPulses")],
                          outdir=args.outdir,
                          num_workers = max(1, multiprocessing.cpu_count() - 2), # Leave a couple of cores free for other work
                        )
# Run Converter
converter(input_dir = args.indir)
converter.merge_files()
