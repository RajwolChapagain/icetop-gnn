from graphnet.data.extractors.icecube.ittruthextractor import ITTruthExtractor
from graphnet.data.extractors.icecube.itfeatureextractor import ITFeatureExtractor
from graphnet.data.dataconverter import DataConverter
from graphnet.data.readers import I3Reader
from graphnet.data.writers import SQLiteWriter

# Your settings
dir_with_files = 'test_data'
outdir = 'test_data_out'
num_workers = 1

# Instantiate DataConverter - exports data from I3 to SQLite
converter = DataConverter(file_reader = I3Reader('data/GeoCalibDetectorStatus_2012.56063_V1_OctSnow_scint.i3'),
                          save_method = SQLiteWriter(),
                          extractors=[ITTruthExtractor(name = "truth_table"), ITFeatureExtractor(name = "features")],
                          outdir=outdir,
                          num_workers=num_workers,
                        )
# Run Converter
converter(input_dir = dir_with_files)
# Merge files (Optional)
converter.merge_files()
