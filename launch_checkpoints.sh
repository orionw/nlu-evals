#!/bin/bash

# for every checkpoint in the folder that doesn't already have results for results/$model_name/xsquad/all_results.json
# run the entry script
for checkpoint in $(ls $1)
# for checkpoint in ${checkpoints[@]}
do
    model_name=$checkpoint
    # strip the / off of $1
    folder=$(echo $1 | sed 's/\/$//')

    # check to see if the checkpoint has results for results/$model_name/xsquad/all_results.json
    if [ ! -f "results/$folder--$model_name/xnli/all_results.json" ]; then
        sbatch run_all.sh "$folder/$model_name"
        echo "Launched $folder/$model_name"
    else
        echo "Skipping $folder/$model_name because it already has results"
    fi
done

