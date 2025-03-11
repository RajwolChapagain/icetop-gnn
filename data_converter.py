from graphnet.data.extractors.icecube.ittruthextractor import ITTruthExtractor
from graphnet.data.extractors.icecube.i3featureextractor import ITFeatureExtractor
from graphnet.data.dataconverter import DataConverter
from graphnet.data.readers import I3Reader
from graphnet.data.writers import SQLiteWriter

dir_with_files = '/home/rchapagain/data/'
outdir = 'data_out/'

# Instantiate DataConverter - exports data from I3 to SQLite
converter = DataConverter(file_reader = I3Reader(dir_with_files + 'gcd/GeoCalibDetectorStatus_2012.56063_V1_OctSnow_scint.i3'),
                          save_method = SQLiteWriter(),
                          extractors=[ITTruthExtractor(name = "truth_table"), ITFeatureExtractor(pulsemap = "OfflineIceTopHLCTankPulses")],
                          outdir=outdir,
                        )
# Run Converter
converter(input_dir = dir_with_files)
converter.merge_files()
