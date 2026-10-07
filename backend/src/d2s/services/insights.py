"""Senior Insights (RQ3): aggregate findings only.

Deliberately no individual scoring: predicting a person's success from personality scores is
ethically sensitive, and the evidence comes from 161 rows of sample data. The findings are for
mentoring, role allocation and structured assessment design.
"""

from d2s.services import store

ETHICS_NOTE = ("Aggregate associations from SAS sample data (161 senior data scientists). Not for "
               "screening or scoring individuals; use for mentoring, role allocation and assessment design.")


def senior() -> dict:
    uni = store.table("rq23/rq3_sds_univariate.csv", "run_rq2_rq3.py")
    models = store.table("rq23/rq3_sds_models.csv", "run_rq2_rq3.py")
    odds = store.table("rq23/rq3_sds_odds_ratios.csv", "run_rq2_rq3.py")
    rules = (store.processed() / "rq23" / "rq3_sds_tree_rules.txt").read_text(encoding="utf-8")
    s = store.summary("rq23/summary.json", "run_rq2_rq3.py")["rq3_sds"]
    return {"note": ETHICS_NOTE, "n": s["audit"]["rows"], "class_counts": s["audit"]["class_counts"],
            "traits": uni.round(4).to_dict("records"), "models": models.round(4).to_dict("records"),
            "odds_ratios": odds.round(4).to_dict("records"), "decision_rules": rules,
            "permutation_test": s["permutation_test"]}


def junior() -> dict:
    uni = store.table("rq23/rq2_jds_univariate.csv", "run_rq2_rq3.py")
    models = store.table("rq23/rq2_jds_models.csv", "run_rq2_rq3.py")
    rules = (store.processed() / "rq23" / "rq2_jds_tree_rules.txt").read_text(encoding="utf-8")
    s = store.summary("rq23/summary.json", "run_rq2_rq3.py")["rq2_jds"]
    return {"n": s["audit"]["rows"], "skills": uni.round(4).to_dict("records"),
            "models": models.round(4).to_dict("records"), "decision_rules": rules,
            "permutation_test": s["permutation_test"]}
