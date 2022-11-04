## How to run run_causal.py:

## causality analysis:
```
cd Socrates
python3 -u ./source/run_causal.py --sepc benchmark/causal/**dataset_name**/**sepc_file.json** --algorithm causal --dataset **dataset_name**
# for Adult Income dataset
e.g. python3 -u ./source/run_causal.py --spec benchmark/causal/census/spec_gender_race.json --algorithm causal --dataset census
# for WikiPedia dataset
e.g. python3 -u source/run_causal.py --spec benchmark/rnn/nnet/wiki_lstm/spec_causal.json --algorithm causal --threshold 0.90 --eps 0.1 --dataset wiki
```
