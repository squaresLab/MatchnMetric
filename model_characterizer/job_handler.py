# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275
import json
import datetime
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
from typing import List, Union
import time
import torch
from datasets import VideoCaTEDataset
from evaluations import Evaluation
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase
from cate_models import *
from perturbations import Perturbation

# Add database to PYTHONPATH to allow for import from lower-level Model definitions
import sys
package_path = os.path.dirname(os.path.realpath(__file__))
print(f"Appending {package_path} to PYTHONPATH")
sys.path.append(package_path)

MAX_MODEL_PER_GPU = 1

# Hack to avoid OSError from opening up multiple processes for some reason?
os.system("ulimit -n 4096")

class JobHandler:
    """This class serves as the front-facing job handler and CLI for the Model Characterizer tool.
       By taking in either a json file detailing datasets/perturbations/models to run or CLI input."""

    def __init__(self):
        """Constructor for Job Handler, sets up logging."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        curr_time = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

    def handle_json_jobs(self, json_paths: List[Union[os.PathLike, str]]):
        """Handles json jobs. Uses supplied json config to set up jobs going over every permutation of
        dataset-perturbation-model using the lists supplied by the user. Stores all actions and results
        in the sqlite3 database. This function also handles parallelization of the mutate-infer jobs. Due
        to current implementation, parallelization is limited by the number of GPUs supplied.

        Arguments:
            json_path: Path to a json job manifest."""

        # CUDA will throw an error if attempting to use from a forked process, this command
        # avoids that by changing the start method multiprocessing uses.
        torch.multiprocessing.set_start_method('spawn')
        job_dict = None
        try:
            job_dict = {}
            for json_path in json_paths:
                with open(json_path, "rb") as f:
                    job_dict.update(json.load(f))
            self.db_path = job_dict["metadata"]["db_root"]
            # Ignored for now, real limit is the number of GPUs.
            # self.num_workers = job_dict["metadata"]["num_workers"]
            self.gpus = job_dict["metadata"]["gpus"]
            self.cache_dset = job_dict["metadata"]["cache-datasets"]
            self.cache_det_img = job_dict["metadata"]["cache-det-images"]


            self.db = ModelCharacterizerDatabase(db_path=self.db_path,
                                                    database=job_dict["metadata"]["database"],
                                                    user=job_dict["metadata"]["user"],
                                                    password=job_dict["metadata"]["password"],
                                                    host=job_dict["metadata"]["host"],
                                                    port=job_dict["metadata"]["port"])
        except FileNotFoundError:
            raise FileNotFoundError("Unable to find the JSON file containing job info.")
        # Main Loop

        # First we need to generate the models, datasets, and perturbations we're using.
        # Done in this order due to foreign key dependencies. Perturbations need to be generated before
        # datasets because datasets have perturbation row IDs as a foreign key.
        models = []
        perturbations = []
        datasets = []

        model_descriptions = job_dict["models"]
        # The "models" field should be a list of dictionaries with 2 keys. For each dictionary,
        # "model_name" is the name of the model to be provided to deeplite torch zoo, and
        # "model_dataset" is the name of the dataset the model is pretrained on.
        for model in model_descriptions:
            try:
                # This is imported from models
                model_class = MODEL_LUT[model["type"]]
            except KeyError as e:
                print(f"Key {model['type']} not found, skipping this model.")
                continue
            print(model_class)
            models.append(model_class(None, model["model_name"], model["model_dictionary"], self.db))


        dataset_descriptions = job_dict["datasets"]
        annotations = []
        input_args = []
        for dataset in dataset_descriptions:
            for perturbation in perturbations:
                datasets.append(VideoCaTEDataset(None,
                                                 db=self.db,
                                                 video_path=dataset["video_path"],
                                                 annotations_path=dataset["annotations_path"],
                                                 perturbation_id=perturbation.id_val))
                annotations.append(dataset["annotations_path"])
        num_workers = len(self.gpus)
        input_args = []
        for i in range(num_workers):
            input_args.append([])  # Add additional queue per GPU.

        i = 0
        t = time.time() # HACK
        
        for dset_idx, dset in enumerate(datasets):
            for idx, model in enumerate(models):
                # i is used to create parallel queues of jobs to hand out between GPUs
                input_args[i].append((model, self.gpus[i], dset, annotations[dset_idx], self.db, t)) # t is included as a HACK
                i = i + 1
                if i == num_workers:
                    i = 0

        # Limiting factor is number of GPUs for now until a more elegant/space-sensitive solution pops up for handling
        # arbitrarily large amounts of datasets at once.

        # If we're caching the datasets, we can pre-compute them all at once.
        if self.cache_dset:
            num_cpu_workers = 24
            work_queues = []
            for i in range(num_cpu_workers):
                work_queues.append([])
            for dset_idx, dset in enumerate(datasets):
                work_queues[dset_idx%num_cpu_workers].append((dset, annotations[dset_idx], self.db))
            with mp.Pool(processes=num_cpu_workers) as pool:
                res = [pool.map(self.handle_mutations, work_queues)]

        print(f"Initializing main work loop with {num_workers} workers. Current execution is limited by number of GPUs.")
        # If num workers is 1, leave it single threaded to allow for easier debugging
        if num_workers == 1:
            for job_list in input_args:
                self.handle_job(job_list)
        with mp.Pool(processes=num_workers) as pool:
            res = [pool.map(self.handle_job, input_args)]
        print(f"This set of jobs took {(time.time() - t)/60} minutes to run!")


    def check_evaluation(self, model_id: int, dataset_id: int, db: ModelCharacterizerDatabase):
        """Checks to see if evaluation already exists; if so, skips the evaluation."""
        attributes = {
            "model_id": model_id,
            "dataset_id": dataset_id
        }
        rows = db.fetch_row_attributes("Evaluations", attributes)
        if len(rows) == 1:
            # Grab 'finished' attribute from DB
            if rows[0][-1] == 1:
                # If evaluation already finished, returned true to skip redoing it.
                return True
            else:
                return False
        else:
            return False

    def handle_mutations(self, dataset_list: List[VideoCaTEDataset]):
        for dataset, annotations_path, db in dataset_list:
            perturb = Perturbation(dataset.perturbation_id, db=db)
            if dataset.cached:
                continue
            else:
                dataset.mutate(dataset.video_path, annotations_path, perturbation=perturb)
                dataset.set_cached(db)
                dataset.set_resolutions(db)

    def handle_job(self, jobs_list: List):
        """Function for handing a dataset perturbation/inferrence job.
        Arguments:
            jobs_list: List of tuples with each tuple containing the components for mutation and inference jobs."""
        for idx, job_args in enumerate(jobs_list):
            model = job_args[0]
            gpu = job_args[1]
            dset = job_args[2]
            annotation = job_args[3]
            db = job_args[4]
            t = job_args[5] # HACK
            if self.check_evaluation(model.id_val, dset.id_val, db):
                print(f"Skipping over previously completed Evaluation on model {model.id_val} and dataset {dset.id_val}")
                continue
            else:
                print(f"""Running job {idx}:\n
                       - model_name: {model.model_name},
                       - dataset: {dset.dataset_name},
                       - perturbation_id: {dset.perturbation_id}\n""")
                success, dataset, workdir = self.handle_dataset(dset, annotation, db)
                if success:
                    self.handle_evaluation(model, workdir, dataset, db, gpu)
                if self.cache_dset is False:
                    shutil.rmtree(workdir, ignore_errors=True)

            print(f"""Successfully run job {idx}:\n
                       - model: {model.model_name},
                       - dataset: {dset.dataset_name},
                       - perturbation_id: {dset.perturbation_id}\n
                    """)
            print(f"Just finished job {idx} with a total time elapsed of {(time.time()-t)/60} minutes!")
    
    def handle_dataset(self, dataset: VideoCaTEDataset, annotation: Union[os.PathLike, str], db: ModelCharacterizerDatabase):
        """Handle dataset perturbation. If self.cache_dset is False, will use a temporary directory that is cleaned up in
        handle_job later.

        Arguments:
            dataset: VideoCaTEDataset object to be perturbed.
            annotation: Path to the annotation file corresponding with this dataset.
            db: Database connection object to use for updating/retrieving.

        Raises:
            OSError: If cached dataset directory does not exist where expected."""

        perturb = Perturbation(dataset.perturbation_id, db=db)
        print("IN HANDLE DATASET")
        if dataset.cached:
            print("USING CACHED DATASET")
            # Dataset is cached, use its ID to find its presumptive path in the DB directory.
            # This currently does not do any validation beyond making sure the directory exists.
            dset_id = dataset.id_val
            expected_dset_path = os.path.join(db.db_root_path, "datasets", str(dset_id))
            if not os.path.exists(expected_dset_path):
                raise OSError(f"Dataset claims to be cached but does not exist at expected path: {expected_dset_path}")
            return True, dataset, expected_dset_path
        if self.cache_dset is False:
            workdir = tempfile.mkdtemp(prefix="cate_model_char_")
            res = dataset.mutate(dataset.video_path, annotations_path=annotation, perturbation=perturb, work_dir=workdir)
            dataset.set_resolutions(db)
            return res, dataset, workdir

    def handle_evaluation(self, model: Model, workdir: Union[os.PathLike, str], dataset: VideoCaTEDataset, db: ModelCharacterizerDatabase, gpu: str):
        """Handle inferrence on a dataset.

        Arguments:
            model: CaTE model wrapper around a torch model to be used on the dataset.
            workdir: Directory where dataset images have been cached.
            dataset: CaTE Dataset object.
            db: Database connection object to use for updating/storing results.
        """
        # Instantiate creates the model, also preps the model for inferrence
        model.instantiate(model.model_name, model.model_dictionary, gpu)
        ev = Evaluation(None, db, model.id_val, dataset.id_val)
        print(f"Running Evaluation {ev.id_val}")
        ev.evaluate(model, workdir, dataset, db)
        ev.set_eval_finished(db)
        ev.write_metadata(model, dataset, db)

    def handle_cli_job(self,
                       video_path: Union[os.PathLike, str],
                       annotations_path: Union[os.PathLike, str],
                       perturbation_path: Union[os.PathLike, str],
                       model: str,
                       model_dataset: str,
                       image_output_path: Union[os.PathLike, str],
                       detections_output_path: Union[os.PathLike, str]):
        """Simple job, no database connections. Meant for one-off experiments.
            dataset_path points to a video file representing a PersonPath dataset.
            annotations_path points to a csv file representing annotations for a PersonPath dataset."""

        perturbation = self.handle_perturbation_json(perturbation_path=perturbation_path)
        dataset = VideoCaTEDataset(id_val=None,
                                   db=None,
                                   video_path=video_path,
                                   annotations_path=annotations_path,
                                   perturbation=perturbation,
                                   output_directory=image_output_path)
        model = Model(id_val=None,
                      db=None,
                      model_name=model,
                      model_dataset=model_dataset)
        print("Mutating dataset")
        dataset.mutate(video_path, annotations_path, perturbation, image_output_path)
        ev = Evaluation(id_val=None, db=None)
        ev.evaluate(model, image_output_path, dataset, db=None, output_dir = detections_output_path)
