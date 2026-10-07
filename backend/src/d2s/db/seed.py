"""Load the course catalogue into the database. (Skill vectors go to ChromaDB: d2s.vector.)"""

import pandas as pd
from sqlalchemy.orm import Session

from d2s.analysis import plan as planlib
from d2s.db.repositories import CourseRepository
from d2s.services import store


def seed_courses(session: Session, catalogue_path=store.CATALOGUE, overwrite: bool = False) -> int:
    repo = CourseRepository(session)
    if repo.count() and not overwrite:
        return 0
    basis = pd.read_csv(catalogue_path).set_index("id")["assumption_basis"].fillna("").to_dict()
    for c in planlib.load_catalogue(catalogue_path):
        repo.upsert(c, assumption_basis=basis.get(c.id) or None)
    return repo.count()
