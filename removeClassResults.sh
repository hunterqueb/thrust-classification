# !/bin/bash



# remove all files from gmat/data/classification except for the python files

# remove all folders from gmat/data/classification 
# DO NOT REMOVE python files
find gmat/data/classification -mindepth 1 -type d -exec rm -rf {} \;
rm gmat/data/classification/*.tex