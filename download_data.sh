# wget -P data/zip http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz
# wget -P data/zip http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz
# wget -P data/zip http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz
# wget -P data/zip http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/splits.zip

# unzip data/zip/splits.zip -d data/raw
tar -xzf data/zip/training.tar.gz -C data/raw
tar -xzf data/zip/validation.tar.gz -C data/raw
tar -xzf data/zip/mmt16_task1_test.tar.gz -C data/raw
# 