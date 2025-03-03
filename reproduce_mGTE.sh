# Run all subsets sequentially
for (( group = 0; group <= 5; group += 1 ))
do
    bash scripts/entry.sh Alibaba-NLP/gte-multilingual-mlm-base results/mgte-mlm-base $group
done

# Get result
python tools/gather_glue.py results/mgte-mlm-base
python tools/gather_xtreme.py results/mgte-mlm-base