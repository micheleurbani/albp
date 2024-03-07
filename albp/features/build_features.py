import gzip
import ecole
import click
import pickle
from pathlib import Path
from datetime import datetime


from models.observation.explore_then_SB import ExploreThenStrongBranch


@click.group()
def cli():
    pass

@cli.command()
@click.option('--source',
              help='The folder from which to read instances.')
@click.option('--destination',
              help='The destination folder where to store data.')
@click.option('--data_max_samples', default=10000,
              help='The number of samples to be collected.')
@click.option('-s', '--seed', default=3478,
              help='Seed value for reproducibility of the experiment.')
def gasse(source: str, destination: str, data_max_samples: int = 10000,
          seed: int = 1234):

    source = Path(source)
    if destination is None:
        fname = datetime.now().strftime('%Y-%m-%d')
        destination = Path('data/interim/gasse/%s' % fname)
    destination.mkdir(exist_ok=True, parents=True)

    instances = ecole.instance.FileGenerator(str(source))

    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 3600,
    }

    # Note how we can tuple observation functions to return complex state
    # information
    env = ecole.environment.Branching(
        observation_function=(
            ExploreThenStrongBranch(expert_probability=0.05),
            ecole.observation.NodeBipartite(),
        ),
        scip_params=scip_parameters,
    )

    # This will seed the environment for reproducibility
    env.seed(seed)

    episode_counter, sample_counter = 0, 0
    # We will solve problems (run episodes) until we have saved enough samples
    while sample_counter < data_max_samples:
        episode_counter += 1

        observation, action_set, _, done, _ = env.reset(next(instances))
        while not done:
            (scores, scores_are_expert), node_observation = observation
            action = action_set[scores[action_set].argmax()]

            # Only save samples if they are coming from the expert (strong
            # branching)
            if scores_are_expert and (sample_counter < data_max_samples):
                sample_counter += 1
                data = [node_observation, action, action_set, scores]
                filename = Path(destination, f"{sample_counter}.pkl")

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)

            observation, action_set, _, done, _ = env.step(action)

        click.echo(f"Episode {episode_counter}, {sample_counter}" + \
                   " samples collected so far")


if __name__ == '__main__':
    cli()