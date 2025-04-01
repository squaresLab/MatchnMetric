# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275
import json
import datetime
import torch
import cv2
import os
from cate_datatypes import CaTEDataType
from perturbations import Perturbation
from cate_models import Model, UltralyticsModel
from datasets import VideoCaTEDataset
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase
from typing import Union
from torchvision.io import read_image
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy

# Enable to save out annotated images alongside evaluations, useful for debugging.
DEBUG_IMAGES = False

class DatasetWrapper(Dataset):
    """Custom Dataset Wrapper used to provide images to the model during the evaluation loop."""

    def __init__(self, image_list: List[torch.tensor]):
        self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class Evaluation(CaTEDataType):
    """Base class for a unified representation of the database/Python object versions of Evaluations. Evaluations
       are the results of running models on perturbed datasets.

       Attributes:
        model_id: Integer representing a model in the database.
        dataset_id: Integer representing a dataset in the database.
        perturb_id: Integer representing a perturbation in the database.
        finished: Boolean indicating if this evaluation has been completed."""

    TABLE = "Evaluations"

    def __init__(self,
                 id_val: int = None,
                 db: ModelCharacterizerDatabase = None,
                 model_id: int = None,
                 dataset_id: int = None):
        kwargs = {
            "model_id": model_id,
            "dataset_id": dataset_id,
        }
        super().__init__(id_val, db, **kwargs)

    def populate_local(self, **kwargs):
        """Constructor assistant method when constructing a CaTE data component without using a database connection.
           Populates object with relevant attributes. Perturbation tries to create the python object and pass the constructor
           parameters to it. Currently assumes a hard-coded resolution.

           Arguments:
            perturb_class: string representing the python import path
            perturb_parameters: dictionary with parameters for the specific mutator. Must match."""
        raise NotImplementedError("This functionality has been deprecated.")

    def populate_by_id(self, id_val: int, db: ModelCharacterizerDatabase):
        """Constructor assistant method when provided an ID corresponding to a row in a CaTE database.
           Arguments:
            id_val: Integer representing the expected row of the item in the database.
            db: Object providing connections to the database where information about this Evaluation is stored. """

        # fetch_row_id returns a list of tuples of len 1 so grab the first one
        # Row is returned as a list without keys; expected format:
        # [id, model_id, dataset_id, date_run, finished]
        db_row = db.fetch_row_id(self.TABLE, id_val)[0]
        if db_row is not None:
            self.id_val = db_row[0]
            self.model_id = db_row[1]
            self.dataset_id = db_row[2]
            self.finished = db_row[-1]

    def check_existance(self, attributes: dict, db: ModelCharacterizerDatabase):
        """Used to check if row matching these attributes already exists in database.
            """
        print(f"ATTRIBUTES: {attributes}")
        rows = db.fetch_row_attributes(self.TABLE, attributes=attributes)
        print(rows)
        if len(rows) > 1:
            raise RuntimeError("Too many perturbations found!")
        if len(rows) == 1:
            row = rows[0]
            return True, row[0]
        else:
            return False, ""

    def insert(self, db: ModelCharacterizerDatabase, model_id, dataset_id):
        date = datetime.datetime.now()
        date_formatted = date.strftime("%m/%d/%Y %H:%M:%S")
        data = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "date_run": date_formatted
        }
        conn = db.get_connection()
        success, row = db.insert(self.TABLE, data, conn)
        if row is not None:
            self.id_val = row[0]
            self.model_id = model_id
            self.dataset_id = dataset_id
        else:
            raise RuntimeError(f"Failed to insert Evaluation with model_id {model_id} and dataset_id {dataset_id}")
        self.finished = False

    def evaluate(self,
                model: Model,
                dataset_dir: Union[os.PathLike, str],
                dataset: VideoCaTEDataset,
                db: ModelCharacterizerDatabase=None,
                output_dir: Union[os.PathLike, str] = None):
        """Iterates over the images and saves the inferences to the output directory."""

        num_workers = 4
        default_batch_size = 64

        if output_dir is None:
            output_dir = os.path.join(db.db_root_path, "evaluations", str(self.id_val))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(os.path.join(output_dir, "detections")):
            os.mkdir(os.path.join(output_dir, "detections"))

        print(f"Made directories for Evaluation {self.id_val}")
        # Grab images
        im_dir = os.path.join(dataset_dir, "images")
        images = os.listdir(im_dir)
        read_image_args = []
        for idx, im in enumerate(sorted(images)):
            read_image_args.append((os.path.join(im_dir, im), idx, model))
        i = 0
        read_images = [None] * len(images)

        # Image loading function to be used in multithreading.
        def image_load(args):
            model = args[2]
            img = read_image(args[0])
            # img = img.unsqueeze(0).permute(0, 3, 1, 2)
            img = img.unsqueeze(0)
            img = model.prep_data_for_inferrence(img)
            read_images[args[1]] = img

        # Single threaded option for debug
        if num_workers == 1:
            for im in read_image_args:
                read_images[im[1]] = read_image[im[0]]
        else:
            with ThreadPool(processes=num_workers) as pool:
                res = [pool.map(image_load, read_image_args)]
        img_size_tensor = read_images[0].size()
        dset = DatasetWrapper(read_images)
        curr_idx = 0
        total_preds = []
        preds = None
        batch = None
        model.prep_model_for_inferrence()
        with torch.no_grad():
            while curr_idx < len(read_images):
                try:
                    # Adjust our batch size if we're at the end of the dataset.
                    if (curr_idx+default_batch_size) > len(read_images)-1:
                        batch_size = (len(read_images) - curr_idx)
                    else:
                        batch_size = default_batch_size
                    # Create the batch
                    batch = torch.stack(dset[curr_idx:curr_idx+batch_size]).squeeze().to(model.device)

                    # When there's a single image in this batch, squeeze() will remove the B dimension from the Tensor
                    # (assuming [B,C,H,W])
                    if len(batch.size()) == 3:
                        batch = batch.unsqueeze(0)

                    print(f"Working on {dataset_dir} with device {batch.device} and curr batch size: {batch_size}")
                    if isinstance(model, UltralyticsModel):
                        print(f"OPERATING ON {model.model.device}")
                        preds = model.model(batch, conf=0.00001, verbose=False)
                    else:
                        preds = model.model(batch)
                    outputs = model.handle_nn_output(preds)
                    total_preds.extend(outputs)

                    curr_idx += batch_size
                    del preds
                    del batch
                    torch.cuda.empty_cache()
                    preds = None
                    batch = None
                except torch.cuda.OutOfMemoryError as e:
                    print(e)
                    # This block handles the case where we try to take up too much memory on the GPU.
                    print("Ran into memory issue")
                    if preds is not None:
                        del preds
                    if batch is not None:
                        del batch
                    default_batch_size = default_batch_size - 2
                    if default_batch_size <= 0:
                        default_batch_size = 1
        i = 0
        # Create directory for output images if we're debugging.
        if DEBUG_IMAGES:
            if not os.path.exists(os.path.join(output_dir, "debug_images")):
                os.mkdir(os.path.join(output_dir, "debug_images"))

        # Iterate over predictions and save out to files.
        for idx, pred_batch in enumerate(total_preds):

            assert len(total_preds) == len(images)
            output_name = f"{str(i)}.txt"
            output_name = output_name.rjust(11, '0')
            numpy.savetxt(os.path.join(output_dir, "detections", output_name), pred_batch)

            # Debug image printing
            if DEBUG_IMAGES:
                img_path = read_image_args[idx][0]
                img = cv2.imread(img_path)
                for pred in pred_batch:
                    x, y, w, h, conf, thing = pred
                    col = (0, 0, 255)
                    cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), col, 2)
                cv2.imwrite(os.path.join(output_dir, "debug_images", f"{idx}.png"),img)

            i += 1
        return True

    def write_metadata(self, model: Model, dataset: VideoCaTEDataset, db: ModelCharacterizerDatabase):
        """Saves a json storing metadata about the evaluation."""
        curr_time = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")
        perturb = Perturbation(dataset.perturbation_id, db=db)
        output_dir = os.path.join(db.db_root_path, "evaluations", str(self.id_val))
        data = {
            "time_completed": curr_time,
            "model": {
                "id_val": model.id_val,
                "model_name": model.model_name,
                "model_dictionary": model.model_dictionary
            },
            "dataset": {
                "id_val": dataset.id_val,
                "name": dataset.dataset_name,
                "perturbation_id": dataset.perturbation_id,
                "perturbation_type": str(perturb.perturb_cls)
            }
        }
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(data, f)

    def set_eval_finished(self, db: ModelCharacterizerDatabase):
        conn = db.get_connection()
        curr = conn.cursor()
        print(f"Setting Evaluation {self.id_val} to finished.")
        curr.execute(f"UPDATE Evaluations SET \"finished\" = 1 WHERE \"id\" = {self.id_val}")
        self.finished=True
        conn.commit()

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as test:
        db = ModelCharacterizerDatabase(test)
        # Get random perturbation
        test_perturb = 'gorgon.mutators.cv_image_rgb.BrightnessContrastMutator'
        parameters = {
            'brightness': 80,
            'contrast': 1.0,
        }
        perturb = Perturbation(None, db, test_perturb, parameters)
        model = Model(None, 'yolo5n', 'coco')
        dset_row_id = db.insert_dataset("coco", "meh", 1)[0]
        print(dset_row_id)
        print(model.id_val)
        eva = Evaluation(None, db, model.id_val, dset_row_id, 1)