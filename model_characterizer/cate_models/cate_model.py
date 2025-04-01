# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275

from cate_datatypes import CaTEDataType
from typing import Union
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase
import datetime
import os
import json


class Model(CaTEDataType):
    """Class for a unified representation of a CaTE model object/database row.
       Attributes:
        row_id - int: ID of the row this model occupies in the database.
        model - Any: reference to a callable ML model. Expected to be able to take in an image and process it.
                    Example usage: pred = self.model(img)
        model_dictionary - dict: Dictionary containing relevant keyword arguments to be used to initialize a particular network.
        """
    TABLE = "Models"

    def __init__(self,
                 id_val: int = None,
                 model_architecture: str = None,
                 model_dictionary: dict = None,
                 db: ModelCharacterizerDatabase = None):
        """Constructor for Models. Given an ID, will find the matching row and populate the rest of the attributes.
           If not given an ID, will use the model_architecture argument to find and save the model using deeplite torch zoo.
           If successful, will insert a row into the database.
           
           Arguments:
            m_id: ID of the row corresponding to the model in the Database.
            model_architecture: String representing the model to be used. Supplied as an argument
                               to deeplite torch zoo's `get_model` function.
            model_dictionary: Dictionary representing keyword arguments used to initialize a model."""

        kwargs = {
            "model_name": model_architecture,
            "model_dictionary": model_dictionary
        }
        super().__init__(id_val, db, **kwargs)

    def populate_local(self, model_architecture: str, model_dataset: str, source: str = 'deeplite'):
        """Constructor assistant method when constructing a CaTE data component without using a database connection.
        Populates object with relevant attributes. Attempts to find model using deeplite_torch_zoo."""
        raise RuntimeError("This function is deprecated.")
 
    def populate_by_id(self, id_val: int, db: ModelCharacterizerDatabase):
        """Constructor assistant method when provided an ID corresponding to a row in a CaTE database.
           Args: 
            id_val: Integer representing the expected row of the item in the database."""
        # Fetch row id returns a list of tuples of len 1 so grab the first one
        db_row = db.fetch_row_id(self.TABLE, id_val)[0]
        if db_row is not None:
            self.id_val = db_row[0] 
            self.model_name = db_row[1]
            self.model_dictionary = json.loads(db_row[3])

    def check_existance(self, attributes: dict, db: ModelCharacterizerDatabase):
        """Given attributes relating to a CaTE model database entry, tries to see if 
           there is one and only one match in the database for the specification.
           Throws an error if there is more than one, """
        rows = db.fetch_row_attributes(table=self.TABLE, attributes=attributes)
        if len(rows) > 1:
            raise RuntimeError("Too many models found!")
        if len(rows) == 1:
            row = rows[0]
            return True, row[0]
        else:
            return False, ""
        
    def instantiate(self):
        """Populates the self.model attribute to allow for inference. """

        raise NotImplementedError("Please call 'instantiate' from a subclass inheriting from Model.")

    def insert(self, db: ModelCharacterizerDatabase, model_name: str, model_dictionary: dict):
        """Inserts model into the database.
           
           Arguments:
            model_dictionary: Dictionary containing all
            db: Database class to use for queries.
            
           Raises:
            RuntimeError: When database insertion fails.("""

        try:
            self.instantiate(model_name, model_dictionary)
        except Exception as e:
            print(f"Error detected during attempted model instantiation. Not adding model {model_name} to database.")
            raise e
        # We instantiate it just to test if it can be instantiated. Delete the model to avoid overpopulating the GPU.
        del self.model
        self._insert(db, model_name, model_dictionary)

    def format_output(self, **kwargs):
        raise NotImplementedError("Please call format_output from a subclass inheriting from Model.")

    def prep_model_for_inferrence(self, **kwargs):
        raise NotImplementedError("Please call prep_for_inference from a subclass inheriting from Model.")

    def _insert(self, **kwargs):
        raise NotImplementedError("Please call _insert from a subclass inheriting from Model.")

    def prep_data_for_inferrence(self, **kwargs):
        """Perform any pre-inference changes on the input data. As an example, this could include image tensor resizing and moving the tensor
            to the same device as the model."""
        raise NotImplementedError("Please call prep_data_for_inferrence from a subclass inheriting from Model.")