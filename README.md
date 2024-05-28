# Reciprocator
Code for the paper "Reciprocal Reward Influence Encourages Cooperation From Self-Interested Agents" (2024).

After cloning this repository, you can install the required dependencies by running:
```
pip install -r requirements.txt
```

To run the experiments, you can use the following command template (with an example for Coins):
```
python run_trainer.py -n reproduce_coins -g coins -c configs/coins.yaml -e 1000 -d cuda:0 -r 8 -dd results
```
