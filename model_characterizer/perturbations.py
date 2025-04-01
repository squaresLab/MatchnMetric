# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275
# make sure no-op is always inserted first into DB
import json
import numpy
from cate_datatypes import CaTEDataType
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase

class no_op():
    """Class representing a perturbation that does nothing, included as a dummy to assist in execution."""
    def __init__(x_res: int, y_res: int, empty_dict: dict):
        pass

    def mutate(self, frame):
        return frame


class Perturbation(CaTEDataType):
    """Base class for a unified representation of the database/Python object versions of Perturbations."""

    TABLE = "Perturbations"

    def __init__(self, 
                 id_val: int = None,
                 perturb_class: str = None,
                 perturb_parameters: dict = None,
                 db: ModelCharacterizerDatabase = None):
        
        kwargs = {
            "perturb_class": perturb_class,
            "perturb_parameters": perturb_parameters
        }

        super().__init__(id_val, db, **kwargs)

    def __call__(self, frame: numpy.ndarray):
        return self.perturbation(frame)
    
    def populate_local(self, perturb_class: str, perturb_parameters: dict):
        """Constructor assistant method when constructing a CaTE data component without using a database connection.
           Populates object with relevant attributes. Perturbation tries to create the python object and pass the constructor
           parameters to it. Currently assumes a hard-coded resolution.
           
           TODO: Custom resolution support.
           Arguments:
            perturb_class: string representing the python import path
            perturb_parameters: dictionary with parameters for the specific mutator. Must match."""
        
        self.perturb_cls = eval(perturb_class)
        try:
            self.perturbation = self.perturb_cls(1080, 1920, **self.perturb_parameters)
        except Exception as e:
            import ipdb; ipdb.set_trace()
    def populate_by_id(self, id_val: int, db: ModelCharacterizerDatabase):
        """Constructor assistant method when provided an ID corresponding to a row in a CaTE database.
           Arguments:
            id_val: Integer representing the expected row of the item in the database."""
        # Fetch row id returns a list of tuples of len 1 so grab the first one
        db_row = db.fetch_row_id(self.TABLE, id_val)[0]
        if db_row is not None:
            self.id_val = int(db_row[0])
            self.perturb_cls = eval(db_row[1])
            self.perturb_parameters = db_row[2]
            self.perturbation = self.perturb_cls(1080, 1920, **self.perturb_parameters)

    def check_existance(self, attributes: dict, db: ModelCharacterizerDatabase):
        """Used to check if row matching these attributes already exists in database. Expected contents of attributes:
           { 'python_path': 'gorgon.mutator.example',
             'attributes' : { dict containing perturbation parameters }
            """
        rows = db.fetch_row_attributes(self.TABLE, attributes=attributes)
        if len(rows) > 1:
            raise RuntimeError("Too many perturbations found!")
        if len(rows) == 1:
            row = rows[0]
            return True, row[0]
        else:
            return False, ""

    def insert(self, db: ModelCharacterizerDatabase, perturb_class: str, perturb_parameters: dict):
        # Sanity check to see if it works
        self.perturb_cls = eval(perturb_class)
        self.perturb_parameters = perturb_parameters
        try:
            self.perturbation = self.perturb_cls(1080, 1920, **self.perturb_parameters)
        except Exception as e:
            import ipdb; ipdb.set_trace()
        data = {
            "perturb_class": perturb_class,
            "perturb_parameters": perturb_parameters
        }
        conn = db.get_connection()
        success, row = ModelCharacterizerDatabase.insert(self.TABLE, data, conn)
        if row is not None:
            self.id_val = row[0]
        else:
            raise RuntimeError("Failed to insert perturbation into database.")
if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as test:
        db = ModelCharacterizerDatabase(test)