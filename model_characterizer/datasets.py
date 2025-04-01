# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275

import multiprocessing as mp
import pandas as pd
from time import sleep
import cv2
import tempfile
import cate_datatypes
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase
from typing import Union
import os
import model_performance_db
from perturbations import Perturbation
import datetime


class CaTEDataset(cate_datatypes.CaTEDataType):
    """Base class for a unified representation of a CaTE Dataset/row in the database."""

    TABLE = "Datasets"

    def _mutate(self,
                video_path: Union[os.PathLike, str],
                annotations_file: Union[os.PathLike, str],
                perturbation: object):
        raise NotImplementedError


class VideoCaTEDataset(CaTEDataset):
    """Class for handling video based datasets in the CaTE database.
       Attributes:
        video_path - Union[os.PathLike, str]: Path to the video file.
        annotations_path - Union[os.PathLike, str]: Path to the annotations file. """

    def populate_local(self, 
                       video_path: Union[os.PathLike, str],
                       annotations_path: Union[os.PathLike, str],
                       perturbation: object,
                       output_directory: Union[os.PathLike, str]):
        """Constructor assistant method when constructing a CaTE data component without using a database connection.
           Populates object with relevant attributes. This is used when mutating a dataset without a database connection."""
        self.id_val = None 
        self.video_path = video_path
        self.annotations_path = annotations_path
        self.perturbation = perturbation
        self.file_directory = output_directory
        if not os.path.exists(self.file_directory):
            os.mkdir(self.file_directory)

    def populate_by_id(self, id_val: int, db: ModelCharacterizerDatabase):
        """Constructor assistant method when provided an ID corresponding to a row in a CaTE database."""
        db_row = db.fetch_row_id(self.TABLE, id_val)[0]
        if db_row is not None:
            self.id_val = db_row[0]
            self.dataset_name = db_row[1]
            # Store path to where this dataset is located on disk for ease of access.
            if db_row[-3] == 1:
                self.cached = True
            else:
                self.cached = False
            self.video_path = db_row[2]
            self.root_path = os.path.join(db.db_root_path, "datasets", str(self.id_val))
            self.perturbation_id = db_row[-4]

    def check_existance(self, attributes: dict, db: ModelCharacterizerDatabase):
        """Used to check if row matching these attributes already exists in database.
           Arguments:
            attributes: Datatype specific attributes used in DB query.
            db: Specific database to use for the query.
           Returns:
            success: Boolean indicating if query was successful.
            row_id: ID corresponding to the database row representing this object."""
        # We construct a dictionary with separate attributes specifically for the search due to janky design.
        search_attr = {}
        dset_basename = os.path.basename(attributes["video_path"])
        search_attr["dataset_name"] = dset_basename
        search_attr["perturbation_id"] = attributes["perturbation_id"]
        rows = db.fetch_row_attributes(self.TABLE, search_attr)
        if len(rows) > 1:
            raise RuntimeError("Too many datasets found!")
        if len(rows) == 1:
            row = rows[0]
            return True, row[0]
        else:
            return False, ""
    
    def insert(self,
               db: ModelCharacterizerDatabase,
               video_path: Union[os.PathLike, str],
               annotations_path: Union[os.PathLike, str],
               perturbation_id: int):
        """Inserts a CaTE dataset into the database. 
           
           Arguments:
            video_path: Absolute path to the video file to be processed into a CaTE dataset.
            annotations_file: Absolute path to the annotations file to be processed alongside the video.
            perturbation_id: ID of the perturbation to be used when mutating the dataset.
            db: Specific database to use for the query. """
        dset_name = os.path.basename(video_path)
        print(f"[DATASET] Inserting new dataset {dset_name} into database.")
        date = datetime.datetime.now()
        date_formatted = date.strftime("%m/%d/%Y %H:%M:%S")
        data = {
            "dataset_name": dset_name,
            "video_path": video_path,
            "date_added": date_formatted,
            "perturbation_id": perturbation_id,
            "cached": 0
        }
        conn = db.get_connection()
        success, row = ModelCharacterizerDatabase.insert(self.TABLE, data, conn)
        if row is not None:
            self.id_val = row[0]
            self.perturbation_id = perturbation_id
            self.video_path = video_path
            self.dataset_name = dset_name
            self.root_path = os.path.join(db.db_root_path, "datasets", str(self.id_val))
            self.cached = False
        else:
            raise RuntimeError(f"Failed to insert {dset_name} to database.")
        print(f"[DATASET] Dataset inserted and given ID: {self.id_val}.")

    def get_by_attributes(self,
                          video_path: Union[os.PathLike, str],
                          perturbation_id: int,
                          db: ModelCharacterizerDatabase):
        """Given video dataset attributes, return the row
           corresponding to that model. Used to populate self once row is found.

           Arguments:
            video_path: Path to the video file.
            annotations_path: Path to the annotations file
            perturbation_id: Row ID of the perturbation used for this dataset."""

        video_basename = os.path.basename(video_path)
        attributes_dict = {
            "dataset_name": video_basename,
            "perturbation_id": perturbation_id 
        }

        rows = db.fetch_row_attributes(self.TABLE, attributes=attributes_dict)

        if len(rows) > 0:
            raise RuntimeError("Too many datasets found!")
        if len(rows) == 1:
            row = rows[-1]
            # self.m_id = row[-1]
            self.id_val = row[0]
            self.dataset_basename = row[0]
            self.root_path = os.path.join(db.db_root_path, "datasets", str(self.id_val))
            self.video_path = row[2]
            self.perturbation_id = row[-2]
            if row[-3] == 0:
                self.cached = False
            else:
                self.cached = True
            return True
        elif len(rows) == 0:
            return False 
    
    def mutate(self, 
                video_path: Union[os.PathLike, str],
                annotations_path: Union[os.PathLike, str],
                perturbation: Perturbation,
                work_dir: Union[os.PathLike, str] = None):
        """Mutates video, saves out each image into a new directory called "images"
           at the location specified by 'out_path'. Also copies annotations over."""

        # Make output directories
        if work_dir is None:
            out_dir = self.root_path
            if not os.path.exists(self.root_path):
                os.mkdir(self.root_path)
        else:
            out_dir = work_dir
        print(f"{out_dir}")
        img_dir = os.path.join(out_dir, "images")
        gt_dir = os.path.join(out_dir, "gt")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        if not os.path.exists(gt_dir):
            os.mkdir(gt_dir)
        # print(self.root_path)


        annotations = pd.read_csv(annotations_path, header=None)
        annotations.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility', 'misc']
        i = 1

        # Start reading the video
        cap = cv2.VideoCapture(video_path)
        print(video_path)
        # Get all the images from the video and hopefully load them into RAM.
        mutator = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if mutator is None:
                self.x_res = frame.shape[1]
                self.y_res = frame.shape[0]
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            img_name = str(i)+".png"
            img_name = img_name.rjust(11, '0')
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, frame)

            # Format GT object
            # Get annotations for the current frame

            frame_annotations = annotations[annotations['frame'] == current_frame]
            frame_annotations.to_csv(os.path.join(gt_dir, str(i)+".csv"))
            i = i + 1
            sleep(0.01)
        print("Done mutating dataset")
        return True, self.id_val
    
    def set_resolutions(self, db: ModelCharacterizerDatabase):
        conn = db.get_connection()
        curr = conn.cursor()
        curr.execute(f"UPDATE Datasets SET \"x_res\" = {self.x_res} WHERE \"id\"={self.id_val}")
        curr.execute(f"UPDATE Datasets SET \"y_res\" = {self.y_res} WHERE \"id\"={self.id_val}")
        conn.commit()

    def set_cached(self, db: ModelCharacterizerDatabase):
        conn = db.get_connection()
        curr = conn.cursor()
        curr.execute(f"UPDATE Datasets SET \"cached\" = 1 WHERE \"id\"={self.id_val}")
        self.cached = True
        conn.commit()

class no_op:
    
    def __call__(frame):
        return frame

# unit tests
if __name__ == "__main__":
    annotation_file = '/home/itar-tswierze/model-robustness-tool/detection-arena/tracking-dataset/person_path_22_data/person_path_22/person_path_22-test/uid_vid_00008.mp4/gt/gt.txt'
    video_file = '/home/itar-tswierze/model-robustness-tool/detection-arena/tracking-dataset/dataset/personpath22/raw_data/uid_vid_00008.mp4'
    test_out_path = '/home/itar-tswierze/model-robustness-tool/testing/'
    with tempfile.TemporaryDirectory() as temp:
        test_db = ModelCharacterizerDatabase(temp)
        perturb_id, _ = test_db.insert_perturbation("no_op", {})
        vid_dset = VideoCaTEDataset(None, video_path=video_file, annotations_path=annotation_file, perturbation_id=1, db=test_db)