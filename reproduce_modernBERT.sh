# Run all subsets sequentially
for (( group = 0; group <= 5; group += 1 ))
do
    bash scripts/entry.sh answerdotai/modernbert-base results/modernbert-base $group
done

# Get result
python tools/gather_glue.py results/modernbert-base
python tools/gather_xtreme.py results/modernbert-base