#!/opt/conda/bin/python

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import seaborn as sns

from afqinsight.datasets import AFQDataset
from neurocombat_sklearn import CombatModel
from plot_formatting import set_size, FULL_WIDTH, TEXT_WIDTH
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from xgboost import XGBRegressor


def predict_age(
    model, X, y, qc, qc_cutoff=0.0, cv=5, scoring=None, score_qc_fail=False
):
    qc_mask = qc >= qc_cutoff
    X_pass = X[qc_mask]
    y_pass = y[qc_mask]

    if score_qc_fail and not all(qc_mask):
        X_fail = X[~qc_mask]
        y_fail = y[~qc_mask]
        model.fit(X_pass, y_pass)
        score_fail = model.score(X_fail, y_fail)
    else:
        score_fail = np.nan

    pass_scores = cross_val_score(model, X_pass, y_pass, cv=cv, scoring=scoring)

    train_size = len(X_pass)
    result_dicts = [
        {
            "cutoff": qc_cutoff,
            "split": cv_idx,
            "score": score,
            "train_size": train_size,
            "qc_fail_score": score_fail,
        }
        for cv_idx, score in enumerate(pass_scores)
    ]

    return pd.DataFrame(result_dicts)


def age_predict_qc_sweep(out_dir, csv_dir, recompute=False, model_type="xgb"):
    n_splits = 5
    if recompute or not op.exists(op.join(csv_dir, "age_predict_qc_sweep.csv")):
        tsv_path = "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv"
        nodes_path = "s3://hbn-afq/derivatives/afq_like_hcp/combined_tract_profiles.csv"

        dataset = AFQDataset(
            fn_nodes=nodes_path,
            fn_subjects=tsv_path,
            dwi_metrics=["dki_fa", "dki_md"],
            index_col="subject_id",
            target_cols=["age", "scan_site_id", "dl_qc_score"],
            label_encode_cols=["scan_site_id"],
        )
        dataset.drop_target_na()

        imputer = SimpleImputer(strategy="median")
        dataset.X = imputer.fit_transform(dataset.X)
        site_ids = dataset.y[:, 1].reshape(-1, 1)

        X_harmonized = CombatModel().fit_transform(
            dataset.X,
            site_ids,
            None,
            None,
        )

        if model_type == "xgb":
            base_model = XGBRegressor(
                objective="reg:squarederror",
                nthread=4,
                colsample_bytree=0.86,
                gamma=0.01,
                learning_rate=0.005,
                max_depth=2,
                min_child_weight=2,
                n_estimators=3000,
                subsample=0.2,
                random_state=0,
            )
        elif model_type == "pcr_lasso":
            pipe_steps = [
                ("scale", StandardScaler()),
                ("pca", PCA()),
                ("lasso", LassoCV()),
            ]
            base_model = Pipeline(steps=pipe_steps)

        model = TransformedTargetRegressor(
            regressor=base_model,
            func=np.log,
            inverse_func=np.exp,
        )

        qc_dfs = [
            predict_age(
                model=model,
                X=X_harmonized,
                y=dataset.y[:, 0],
                qc=dataset.y[:, 2],
                qc_cutoff=cutoff,
                cv=RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=0),
            )
            for cutoff in tqdm(np.linspace(0, 1, 20, endpoint=False))
        ]

        qc_df = pd.concat(qc_dfs)
        qc_df.to_csv(op.join(csv_dir, "age_predict_qc_sweep.csv"))
    else:
        qc_df = pd.read_csv(op.join(csv_dir, "age_predict_qc_sweep.csv"))

    qc_df["Training set size"] = (n_splits - 1) / n_splits * qc_df["train_size"]
    colors = plt.get_cmap("tab10").colors

    fig, ax0 = plt.subplots(figsize=set_size(width=0.5 * TEXT_WIDTH))

    line_width_reduction = 0.5
    linewidth = mpl.rcParams["lines.linewidth"]
    with mpl.rc_context({"lines.linewidth": line_width_reduction * linewidth}):
        _ = sns.regplot(
            data=qc_df,
            x="cutoff",
            y="score",
            x_estimator=np.mean,
            truncate=False,
            color=colors[0],
            fit_reg=False,
            scatter_kws={"s": 11},
            label="Age prediction R^2 (left axis)",
            ax=ax0,
        )
    _ = ax0.set_xticks(np.linspace(0, 1, 11))
    ax1 = ax0.twinx()
    _ = sns.lineplot(
        data=qc_df,
        x="cutoff",
        y="Training set size",
        lw=1,
        ax=ax1,
        color=colors[1],
        label="Training set size (right axis)",
    )
    handles1, _ = ax1.get_legend_handles_labels()
    ax1.get_legend().remove()
    handles0, _ = ax0.get_legend_handles_labels()
    handles = handles0 + handles1
    labels = [r"$R^2$", "Training size"]
    _ = ax0.legend(handles, labels, loc="lower center")

    _ = ax0.set_xlabel("QC cutoff")
    _ = ax1.set_xlabel("QC cutoff")
    _ = ax0.set_ylabel(r"Age prediction $R^2$")

    fig.savefig(op.join(out_dir, "qc_sweep.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_scores_dir",
        help="Figure directory.",
    )

    parser.add_argument(
        "fig_dir",
        help="Figure directory.",
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute the cross-validated scores for each QC cutoff.",
    )

    args = parser.parse_args()

    with plt.style.context("/tex.mplstyle"):
        age_predict_qc_sweep(
            out_dir=args.fig_dir,
            csv_dir=args.csv_scores_dir,
            recompute=args.recompute,
        )
