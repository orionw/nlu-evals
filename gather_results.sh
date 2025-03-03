#!/bin/bash

# for direct subfolders in results/ 
for model_folder in results/*/; do
    # if it is a folder and not a symlink
    if [ -d "$model_folder" ] && [ ! -L "$model_folder" ]; then
        echo "Processing $model_folder"
        python tools/gather_glue.py $model_folder
        python tools/gather_xtreme.py $model_folder
    fi
done
