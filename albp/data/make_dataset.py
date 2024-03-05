# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dotenv import find_dotenv, load_dotenv

from albp.data.salbp import SALBP


logger = logging.getLogger(__name__)

RAW_DATA = Path('data', 'raw', 'albp-datasets')
RAW_DATA.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA = Path('data', 'processed', 'albp-datasets')
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option("-f", "--force", is_flag=True, show_default=True, default=False,
              help="Force rewriting the model(s).")
def main(force):
    """
    Write '.lp' models in the '../processed/lp' folder.

    Check the presence of all MILP instances, and, in case they are missing, it
    runs data processing scripts to turn raw data from (../raw/albp-datasets)
    into cleaned data ready to be analyzed (saved in ../processed/lp).

    """
    # SALBP-1993
    dataset = 'SALBP-1993'
    logger.info(f"Start writing models in {dataset}.")
    problems = Path(RAW_DATA, dataset)
    for i in problems.glob('*.IN2'):
        instance = i.name.replace('.IN2', '')
        path = Path(PROCESSED_DATA, dataset, instance + '.lp')
        if (path.exists() and force) or not path.exists():
            generate_model(dataset=dataset, instance=instance,
                            cycle_time=None)
    # SALBP-2013
    dataset = 'SALBP-2013'
    logger.info(f"Start writing models in {dataset}.")
    problems = Path(RAW_DATA, dataset)
    problem_sizes = ['20', '50', '50-permuted', '100', '1000']
    data = []
    for size in problem_sizes:
        ppath = Path(problems, size)
        for i in ppath.glob('*.alb'):
            instance = i.name.replace('.alb', '')
            path = Path('lp', dataset, size, instance + '.alb')
            if (path.exists() and force) or not path.exists():
                data.append(
                    (dataset, instance, size, None)
                )

    with Pool(processes=cpu_count()) as p:
        p.starmap(generate_model, data)

def generate_model(
        dataset: str,
        instance: str,
        size: str = None,
        cycle_time: int = None,
        option: str = None,
        solver_id: str = "SCIP"
    ):

    logger.info(f"Generate model {instance} from {dataset}.")

    params = {
        'dataset': dataset,
        'instance': instance,
        'c': cycle_time,
        'type': 1,
    }

    if dataset == "SALBP-1993":
        problem = SALBP(params, RAW_DATA, PROCESSED_DATA)
    elif dataset == "SALBP-2013":
        params['size'] = size
        problem = SALBP(params, RAW_DATA, PROCESSED_DATA)
    elif dataset == "ARALBP":
        pass
    elif dataset == "PALBP":
        pass

    model = problem.write_model()
    with open(Path(problem.model_folder, instance + '.lp',  ), 'w') as f:
        f.write(model.ExportModelAsLpFormat(obfuscated=False))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
