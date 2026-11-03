import random

import yaml

import Algorithm

class CCEA(Algorithm.CoevolutionaryAlgorithm):
    def __init__(self, alg_config_filename, domain_name, rover_config_filename, data_filename):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)
        if self.num_objs != 1:
            raise ValueError(f"[CCEA] Expected 1 objective, but the environment has {self.num_objs}.")

    def _read_config(self):
        """Read and load CCEA configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[CCEA]: YAML config read.')
        self._load_config()

    def evolve(self, gen=0, traj_write_freq=100):
        """Evolve one generation of a cooperative coevolutionary algorithm.

        Each rover has its own subpopulation. Teams are formed by index after
        shuffling and scored on the single global reward. Each policy is then
        credited with a difference evaluation: reward of the logged trajectory
        minus the reward of that same trajectory with that rover removed. The
        top half of each subpopulation, ranked by that credit, is kept and the
        rest is refilled by binary tournament, SBX, and mutation.
        """
        for subpop in self.pop:
            random.shuffle(subpop)

        # One difference evaluation per policy, aligned with its subpopulation index.
        difference_evals = [[] for _ in range(self.team_size)]
        team_fitnesses = []
        best_fitness = None
        best_trajectory = None

        # Assemble teams from the subpopulations and evaluate them
        for eval_idx in range(self.pop_size):
            team_policy = [self.pop[i][eval_idx] for i in range(self.team_size)]
            trajectory, fitness_dict = self.interface.rollout(team_policy)
            self.glob_eval_counter += 1
            if len(fitness_dict) != self.num_objs or 0 not in fitness_dict:
                raise ValueError(f"[CCEA] Expected objective 0, but got {fitness_dict}.")
            fitness = float(fitness_dict[0])
            team_fitnesses.append(fitness)
            if best_fitness is None or fitness > best_fitness:
                best_fitness = fitness
                best_trajectory = trajectory

            self.data_logger.add_data(key='gen', value=gen)
            self.data_logger.add_data(key='id', value=self.glob_eval_counter)
            self.data_logger.add_data(key='fitness', value=[fitness])
            if gen == self.num_gens - 1 or gen % traj_write_freq == 0:
                self.data_logger.add_data(key='trajectory', value=trajectory)
            else:
                self.data_logger.add_data(key='trajectory', value=None)
            self.data_logger.write_data()

            # Store the full team reward and then subtract the reward of each counterfactual team
            # with one rover removed.
            global_from_traj = float(self.interface.evaluate_trajectory(trajectory)[0])
            for agent_idx in range(self.team_size):
                counterfactual = [trajectory[i] for i in range(self.team_size) if i != agent_idx] # skip rover `agent_idx` to create a counterfactual team
                cf_reward = float(self.interface.evaluate_trajectory(counterfactual)[0]) # evaluate the counterfactual team
                difference_evals[agent_idx].append(global_from_traj - cf_reward)

        mean_fitness = sum(team_fitnesses) / len(team_fitnesses)
        flat_differences = [d for subpop_d in difference_evals for d in subpop_d]
        mean_difference = sum(flat_differences) / len(flat_differences)
        max_difference = max(flat_differences)
        print(f"[CCEA] gen {gen}  mean {mean_fitness:.3f}  best {best_fitness:.3f}  mean_d {mean_difference:.3f}  max_d {max_difference:.3f}")
        if gen == self.num_gens - 1:
            self._report_best_team(best_trajectory, best_fitness)

        # Fill the subpopulations with the top half of the teams, ranked by credit
        # and the rest with binary tournament->SBX->mutation.
        # The SBX (Simulated Binary Crossover) is a crossover operator
        for subpop_idx in range(self.team_size):
            ranked = sorted(range(self.pop_size), key=lambda i, s=subpop_idx: difference_evals[s][i], reverse=True)
            parents = [self.pop[subpop_idx][i] for i in ranked[:self.pop_size // 2]]
            offspring_set = []
            while len(parents) + len(offspring_set) < self.pop_size:
                # Binary tournament selection
                idx1, idx2 = random.sample(range(len(parents)), 2)
                parent1 = parents[min(idx1, idx2)]
                idx1, idx2 = random.sample(range(len(parents)), 2)
                parent2 = parents[min(idx1, idx2)]
                # SBX crossover
                offspring1, offspring2 = self.utils.SBX(parent1, parent2)
                # Mutation
                offspring1.mutate()
                offspring2.mutate()
                offspring_set.extend([offspring1, offspring2])
            self.pop[subpop_idx] = (parents + offspring_set)[:self.pop_size]

    def _report_best_team(self, trajectory, fitness):
        """Print where the best final team finished, and which POIs it covered."""
        if trajectory is None:
            return
        final_positions = [[round(float(coord), 2) for coord in agent_traj[-1]['position']] for agent_traj in trajectory]
        print(f"[CCEA] best final team fitness {fitness:.3f}")
        print(f"[CCEA] best final team last logged positions {final_positions}")
        for poi in self.interface.rover_env.pois:
            covered_steps = 0
            for t in range(len(trajectory[0])):
                positions = [trajectory[a][t]['position'] for a in range(self.team_size)]
                observers = 0
                for loc in positions:
                    distance = sum((loc_i - poi_i) ** 2 for loc_i, poi_i in zip(loc, poi.location)) ** 0.5
                    if distance < poi.radius:
                        observers += 1
                if observers >= poi.coupling:
                    covered_steps += 1
            print(f"[CCEA] POI at {poi.location} covered on {covered_steps} logged steps")
