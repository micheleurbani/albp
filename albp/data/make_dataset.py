# -*- coding: utf-8 -*-
import gzip
import click
import pickle
import logging
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dotenv import find_dotenv, load_dotenv

from albp.data.salbp import SALBP
from albp.data.palbp import PALBP


logger = logging.getLogger(__name__)

RAW_DATA = Path('data', 'raw', 'albp-datasets')
RAW_DATA.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA = Path('data', 'processed', 'albp-datasets')
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

AVAILABLE_PROBLEMS = ['SALBP-1993', 'SALBP-2013', 'PALBP', 'TALBP', 'ARALBP',
                      'VWALBP', 'ALWABP']


@click.command()
@click.option("-f", "--force", is_flag=True, show_default=True, default=False,
              help="Force rewriting the model(s).")
@click.option("-p", "--problem", show_default=True, default='all',
              help="Write problems for a specific datasets.")
def main(force, problem):
    """
    Write '.lp' models in the '../processed/lp' folder.

    Check the presence of all MILP instances, and, in case they are missing, it
    runs data processing scripts to turn raw data from (../raw/albp-datasets)
    into cleaned data ready to be analyzed (saved in
    ../processed/albp-datasets).

    """
    # check problem name for typos
    if problem not in AVAILABLE_PROBLEMS:
        raise ValueError('Problem name is not recognized.')
    data = []
    # SALBP-1993
    if (problem == 'all') or (problem == 'SALBP-1993'):
        dataset = 'SALBP-1993'
        logger.info(f"Start writing models in {dataset}.")
        problems = Path(RAW_DATA, dataset)
        for i in problems.glob('*.alb'):
            instance = i.name.replace('.alb', '')
            path = Path(PROCESSED_DATA, dataset, instance + '.lp')
            if (path.exists() and force) or not path.exists():
                data.append((dataset, instance, None, None))

    # SALBP-2013
    if (problem == 'all') or (problem == 'SALBP-2013'):
        dataset = 'SALBP-2013'
        logger.info(f"Start writing models in {dataset}.")
        problems = Path(RAW_DATA, dataset)
        problem_sizes = [
            '20',
            '50',
            '50-permuted',
            '100',
            '1000'
        ]
        for size in problem_sizes:
            ppath = Path(problems, size)
            for i in ppath.glob('*.alb'):
                instance = i.name.replace('.alb', '')
                path = Path(PROCESSED_DATA, dataset, size, instance + '.lp')
                if (path.exists() and force) or not path.exists():
                    data.append((dataset, instance, size, None))

    # PALBP
    if (problem == 'all') or (problem == 'PALBP'):
        dataset = 'PALBP'
        logger.info(f"Start writing models in {dataset}.")
        problems = Path(RAW_DATA, dataset)
        # only instances in the original paper by Goken and Scholl & Boysen are
        # considered
        C = pd.read_csv(Path(problems, 'cycle-times.csv'))
        for instance in C.problem.unique():
            # retrieve cycle times
            cc = C[C.problem == instance.lower()]
            # check if the instance folder is upper or lowercase
            if Path(RAW_DATA, dataset, instance.upper()).is_dir():
                instance = instance.upper()
            for c in cc.C:
                path = Path(PROCESSED_DATA, dataset, instance + f'_{c}.lp')
                if (path.exists() and force) or not path.exists():
                    data.append((dataset, instance, None, c))

    with Pool(processes=cpu_count()) as p:
        p.starmap(generate_model, data)

def generate_model(
        dataset: str,
        instance: str,
        size: str = None,
        cycle_time: int = None,
    ):

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
    elif dataset == "PALBP":
        problem = PALBP(params, RAW_DATA, PROCESSED_DATA)
    elif dataset == "ARALBP":
        pass

    # create pyscipopt.Model
    model = problem.write_model()

    if dataset == "PALBP":
        instance = instance + f"_{cycle_time}"

    logger.info(f"Generate model {instance} from {dataset}.")

    # write model to .lp file
    model.writeProblem(Path(problem.model_folder, instance + '.lp'))
    # dump problem data to pickle file
    with gzip.open(Path(problem.model_folder, instance), 'wb') as f:
        pickle.dump(model.data, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
