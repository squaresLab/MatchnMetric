# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275
from time import sleep
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase


class CaTEDataType:
    """Base class for CaTE datatypes. Subclasses are intended to represent an interface between the files/components
       of each datatype instance and their database representation.
       Attributes:
        TABLE: String representing which table the object connects to."""
    
    TABLE: str = ""

    def __init__(self, id_val: int = None, db: ModelCharacterizerDatabase = None, **kwargs):
        if id_val is None:
            if db is None:
                # No DB provided so this is a local only data object without DB representation.
                self.populate_local(**kwargs)
            elif db is not None:
                # DB provided so this object might already exist in the database.
                success, ret_id = self.check_existance(kwargs, db)
                if not success:
                    # No matching row found, we need to insert a new one.
                    self.insert(db, **kwargs)
                else:
                    # Matching row found in database, we can grab the attributes from that instead.
                    self.populate_by_id(ret_id, db) 
        else:
            self.populate_by_id(id_val, db)
        
    def populate_local(self, **kwargs):
        """Constructor assistant method when constructing a CaTE data component without using a database connection.
           Populates object with relevant attributes. Expected arguments changes based on specific datatype."""
        raise NotImplementedError("populate_local is meant to be overwritten by specific datatype subclasses.")

    def populate_by_id(self, id_val: int, db: ModelCharacterizerDatabase):
        """Constructor assistant method when provided an ID corresponding to a row in a CaTE database."""
        raise NotImplementedError("populate_by_id is meant to be overwritten by specific datatype subclasses.")

    def check_existance(self, attributes: dict, db: ModelCharacterizerDatabase):
        """Used to check if row matching these attributes already exists in database.
            Arguments:
                attributes: Datatype specific attributes used in DB query.
                db: Specific database to use for the query.
            Returns:
                success: Boolean indicating if query was successful.
                row_id: ID corresponding to the database row representing this object."""
        raise NotImplementedError("check_existance is meant to be overwritten by specific datatype subclasses.")
 
    def insert(self, attributes: dict, db: ModelCharacterizerDatabase):
        """Inserts a CaTE datatype into the database. 
           Arguments:
            db: Database connection to use
            attributes: Column values to insert into database."""
        raise NotImplementedError("insert is meant to be overwritten by specific datatype subclasses.")
