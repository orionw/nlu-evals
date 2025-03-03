# Run all subsets sequentially
for (( group = 0; group <= 5; group += 1 ))
do
    bash scripts/entry.sh FacebookAI/xlm-roberta-base results/xlm-roberta-base $group
done

# Get result
python tools/gather_glue.py results/xlm-roberta-base
python tools/gather_xtreme.py results/xlm-roberta-base