"""Helpers shared by test modules (pytest puts this directory on sys.path)."""


def artifacts_ready() -> bool:
    from d2s.services import store

    try:
        store.artifact_path("hike_model.pkl")
        store.table("rq1/triangulation_rq1_rq2.csv")
        return True
    except FileNotFoundError:
        return False
