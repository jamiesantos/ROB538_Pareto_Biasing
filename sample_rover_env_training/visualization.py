import argparse
import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import yaml


def load_trajectory(data_path, generation, solution_id):
    """Load a specific trajectory from the saved CSV data."""

    data_path = Path(data_path)

    # Allow either a CSV file or a directory containing savedata.csv
    if data_path.is_dir():
        data_path = data_path / "savedata.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if int(row["gen"]) == generation and int(row["id"]) == solution_id:
                trajectory = row["trajectory"]
                break
        else:
            raise ValueError(
                f"No trajectory found for generation={generation}, "
                f"id={solution_id}"
            )

    # Convert NumPy representations into Python literals.
    trajectory = re.sub(
        r"np\.(?:float|int)\d*\(([^()]*)\)",
        r"\1",
        trajectory,
    )

    trajectory = re.sub(
        r"array\((\[[^\]]*\]),\s*dtype=[^)]+\)",
        r"\1",
        trajectory,
    )

    return ast.literal_eval(trajectory)


def load_environment(config_path):
    """Load domain dimensions and POI configuration."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    environment = config["Environment"]

    dimensions = environment["dimensions"]
    pois = environment["pois"]

    return dimensions, pois


def visualize(trajectory, dimensions, pois, generation, solution_id):

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw environment boundaries
    ax.set_xlim(0, dimensions[0])
    ax.set_ylim(0, dimensions[1])
    ax.set_aspect("equal")

    # Draw POIs and their observation radii
    for i, poi in enumerate(pois):
        x, y = poi["location"]
        radius = poi["radius"]

        circle = Circle(
            (x, y),
            radius,
            alpha=0.2,
            label=f"POI {i} radius",
        )
        ax.add_patch(circle)

        ax.scatter(
            x, y,
            marker="*",
            s=150,
            zorder=3,
            label=f"POI {i} (objective {poi['obj']})",
        )

    # Draw rover trajectories
    for agent_id, agent_trajectory in enumerate(trajectory):

        positions = [
            timestep["position"]
            for timestep in agent_trajectory
        ]

        x = [position[0] for position in positions]
        y = [position[1] for position in positions]

        ax.plot(
            x, y,
            marker=".",
            markersize=4,
            label=f"Rover {agent_id}",
        )

        # Starting position
        ax.scatter(x[0], y[0], marker="o", s=70, zorder=4)

        # Ending position
        ax.scatter(x[-1], y[-1], marker="X", s=70, zorder=4)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"Rover Trajectories | Generation {generation}, ID {solution_id}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():

    parser = argparse.ArgumentParser(
        description="Visualize rover trajectories."
    )

    parser.add_argument("data", help="Path to trajectory CSV or data directory")
    parser.add_argument("generation", type=int, help="Generation number")
    parser.add_argument("id", type=int, help="Solution ID")

    parser.add_argument(
        "--config",
        default="config/ccea/4ag_2poi_env.yaml",
        help="Path to rover environment YAML",
    )

    args = parser.parse_args()

    trajectory = load_trajectory(
        args.data,
        args.generation,
        args.id,
    )

    dimensions, pois = load_environment(args.config)

    visualize(
        trajectory,
        dimensions,
        pois,
        args.generation,
        args.id,
    )


if __name__ == "__main__":
    main()