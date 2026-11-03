# Single-objective rover coevolution

Four rovers start at the centre of a 20×20 map. Two points of interest each provide a reward only while at least two rovers are inside. A cooperative coevolutionary algorithm gives every rover its own population and credits each policy with a difference evaluation: the team reward minus the reward of the same trajectory with that rover removed. The RL algorithm, TD3, is also implemented with the commands to run either learning algorithm below.

Requires Python with `numpy`, `pyyaml`, and `torch`.

CCEA Command:
```bash
python main.py ccea rover results/ccea_4ag \
  config/ccea/4ag_2poi_ccea.yaml \
  config/ccea/4ag_2poi_env.yaml \
  3 ccea_diff 1000
```

TD3 Command:
```bash
python main.py td3 rover results/td3_4ag \
  config/td3/4ag_2poi_td3.yaml \
  config/ccea/4ag_2poi_env.yaml \
  3 td3_run 100
```
