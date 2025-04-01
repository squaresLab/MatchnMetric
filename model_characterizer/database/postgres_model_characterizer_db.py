# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, cursor
from psycopg2.extras import Json, DictCursor
from typing import Union
import os
import json

"""PostgreSQL implementation of the Model Characterizer database. This move was done to
support a higher number of concurrent and repeated operations in addition to faster query performance
for more advanced queries.
"""


class ModelCharacterizerDatabase:

    def __init__(self,
                 db_path: Union[os.PathLike, str],
                 database: str,
                 user: str,
                 password: str,
                 host: str,
                 port: int):
        """
        Creates the ModelCharacterizerDatabase object and connects to the relevant database.
        Creates the database in PostgreSQL if required.
        Arguments:
            db_path: Root path where results from dataset mutation and evaluation will be stored.
            database: Name of the database to connect to within Postgres
            user: PostgreSQL username to use to connect to the database
            password: Password for the PostgreSQL user
            host: Hostname for PostgreSQL, most likely just 'localhost' unless we've managed to set up a server for PostgreSQL.
            port: Port to connect to for PostgreSQL access, default is 5432
        """
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_root_path = db_path
        try:
            conn = psycopg2.connect(database=self.database,
                                    user=self.user,
                                    password=self.password,
                                    host=self.host,
                                    port=self.port)
        except psycopg2.OperationalError as e:
            # There has to be a better way to do this
            if f"\"{database}\" does not exist" in str(e):
                print(f"Desired database {database} not found, creating database {database}.")
                # Connect to the "postgres" database so we can make our new DB.
                try:
                    conn = psycopg2.connect(database="postgres",
                                            user=self.user,
                                            password=self.password,
                                            host=self.host,
                                            port=self.port)
                    # This isolation level allows us to make the database in a single statement.
                    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                    curr = conn.cursor()
                    curr.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
                    curr.close()
                    conn.close()
                    # Reconnect, this time to the correct database.
                    conn = psycopg2.connect(database=self.database,
                                            user=self.user,
                                            password=self.password,
                                            host=self.host,
                                            port=self.port)
                    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                    curr = conn.cursor()

                    self.initialize_database(curr)
                except Exception as e:
                    raise Exception(e)
            else:
                print("Unknown error encountered!")
                raise Exception(e)

    def get_connection(self):
        """Returns a connection to the database using this object's internal variables.

        Arguments:
            None
        Returns:
            psycopg2.connection: Connection object to the database corresponding to this object.
        Raises:
            psycopg2.OperationalError: Will raise if not able to return a connection to the database.
        """
        try:
            return psycopg2.connect(database=self.database,
                                    user=self.user,
                                    password=self.password,
                                    host=self.host,
                                    port=self.port)
        except psycopg2.OperationalError as e:
            raise e

    # Normally I prefer the full import path but psycopg2 does some irritating import magic with extensions
    # full path for the cursor is psycopg2.extensions.cursor
    def initialize_database(self, curr: cursor):
        """First-time setup for a new database. Creates tables as well as necessary directory structure.
        Args:
            curr: Cursor in the current psycopg2 session.
        """

        # Used to store the file path
        curr.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS Metadata (
                                file_path VARCHAR(100) NOT NULL,
                                comment VARCHAR(1000)
                            )"""))

        curr.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS Models (
                                id SERIAL PRIMARY KEY,
                                model_name VARCHAR(100) NOT NULL,
                                date_added TIMESTAMP NOT NULL,
                                model_dictionary VARCHAR(500),
                                UNIQUE (model_name, model_dictionary)
                            )"""))

        curr.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS Perturbations (
                                id SERIAL PRIMARY KEY,
                                perturb_class VARCHAR(100) NOT NULL,
                                perturb_parameters jsonb NOT NULL,
                                UNIQUE (perturb_class, perturb_parameters)
                            )"""))

        curr.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS Datasets (
                                id SERIAL PRIMARY KEY,
                                dataset_name VARCHAR(100) NOT NULL,
                                video_path VARCHAR(300) NOT NULL,
                                date_added TIMESTAMP NOT NULL,
                                perturbation_id INTEGER REFERENCES Perturbations(id),
                                cached INTEGER NOT NULL,
                                x_res INTEGER,
                                y_res INTEGER,
                                UNIQUE (dataset_name, perturbation_id)
                            )"""))

        curr.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS Evaluations (
                                id SERIAL PRIMARY KEY,
                                model_id INTEGER REFERENCES Models(id),
                                dataset_id INTEGER REFERENCES Datasets(id),
                                date_run TIMESTAMP NOT NULL,
                                finished INTEGER,
                                UNIQUE (model_id, dataset_id)
                            )"""))

        os.mkdir(os.path.join(self.db_root_path, "datasets"))
        os.mkdir(os.path.join(self.db_root_path, "evaluations"))
        os.mkdir(os.path.join(self.db_root_path, "models"))

    # CRUD
    # Inserts
    @staticmethod
    def insert(table: str, data: dict, conn: psycopg2.extensions.connection):
        """Generic insert method for the database, unrolls the data dict into the relevant keys and data.
        Arguments:
            table: The table to be inserted into.
            data: Dictionary containing the column names and row values to be inserted into the table.
            curr: psycopg2 cursor to use for the insertion operation."""

        curr = conn.cursor()
        insert_query = f"INSERT INTO {table} ( "
        table_columns = ""
        row_data = ""
        for key in data:
            table_columns = table_columns + f"{key}, "
            if type(data[key]) is dict:
                row_data = row_data + f"{Json(data[key])}, "
            else:
                row_data = row_data + f"'{data[key]}', "
        # Remove trailing spaces/commas
        table_columns = table_columns[:-2]
        row_data = row_data[:-2]
        insert_query = insert_query + table_columns
        insert_query = insert_query + " ) "
        insert_query = insert_query + " values ( " + row_data + f" ) RETURNING id, {table_columns}"
        try:
            curr.execute(sql.SQL(insert_query))
            row = curr.fetchone()  # Get the row we just inserted
            conn.commit()
            return True, row
        except psycopg2.OperationalError as e:
            print(f"Encountered error on insert into {table} with query:\n{insert_query}")
            print(f"Error:\n{e}")
            conn.rollback()
            return False, None

    def fetch_row_attributes(self, table: str, attributes: dict):
        """Generic query wrapper for using a select statement to find a row in an
        arbitrary table with arbitrary arguments.
        Arguments:
            table: Table to run the select query on.
            attributes: Dictionary of column names and values to use when searching.
        Returns:
            List of all rows matching the select criteria."""
        conn = self.get_connection()
        sel_query = f"SELECT * FROM {table} WHERE "
        # Construct a query using all provided attributes.
        for idx, key in enumerate(attributes.keys()):
            if isinstance(attributes[key], dict):
                # Json wrapper for psycopg2 to allow for use of dictionaries in queries.
                attribute_str = Json(attributes[key])
            elif isinstance(attributes[key], int):
                attribute_str = f"{attributes[key]}"
            else:
                attribute_str = f"'{str(attributes[key])}'"
            sel_query = sel_query + f"\"{key}\"={attribute_str}"
            if idx != len(attributes.keys())-1:
                sel_query = sel_query + " AND "
        try:
            curr = conn.cursor()
            curr.execute(sel_query)
            result = curr.fetchall()
            if result is None:
                return []
            else:
                return result
        except Exception as e:
            raise Exception(e)

    def fetch_row_id(self, table: str, row_id: int):
        """Given an ID value, get the row matching that ID value from the table.

        Arguments:
            table: Table to query into
            row_id: Integer indicating the id value to search for
        Returns:
            List of rows found by the search. Almost all of the time, this will be
            of length 1.
        """
        try:
            conn = self.get_connection()
            curr = conn.cursor()
            curr.execute(sql.SQL(f"SELECT * FROM {table} where id={row_id}"))
            return curr.fetchall()
        except Exception as e:
            # TODO Logging
            raise Exception(e)


if __name__ == "__main__":
    db_dict = {
            "user": "itar-tswierze",
            "password": "password",
            "database": 'testing_time',
            "host": 'localhost',
            "port": 5432,
            "file_location": "/mnt/drive/test_model_characterizer"
        }

    database = ModelCharacterizerDatabase(db_dict["file_location"], db_dict["database"], db_dict["user"], db_dict["password"], db_dict["host"], db_dict["port"])
    conn = database.get_connection()
    curr = conn.cursor()
