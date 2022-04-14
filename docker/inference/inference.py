#!/opt/conda/bin/python

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import seaborn as sns

from afqinsight.datasets import AFQDataset
from afqinsight.plot import plot_tract_profiles
from groupyr.transform import GroupExtractor
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

BBOX = dict(
    linewidth=1,
    facecolor="white",
    edgecolor="black",
    boxstyle="round,pad=0.25",
)

TEXT_KWARGS = dict(
    x=-0.15,
    y=1,
    ha="center",
    va="center",
    zorder=100,
    fontweight="bold",
    bbox=BBOX,
    alpha=1.0,
    fontsize=9,
)


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


def plot_slfl_profiles():
    dataset = AFQDataset.from_files(
        fn_nodes="s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/afq/combined_tract_profiles.csv",
        fn_subjects="s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        target_cols=["dl_qc_score", "scan_site_id"],
        dwi_metrics=["dki_fa", "dki_md"],
        label_encode_cols=["scan_site_id"],
        index_col="subject_id",
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

    ge = GroupExtractor(
        select="SLF_L", groups=dataset.groups, group_names=dataset.group_names
    )

    X_harmonized = ge.fit_transform(X_harmonized)
    group_names = [grp for grp in dataset.group_names if "SLF_L" in grp]
    groups = [np.arange(100), np.arange(100, 200)]

    metrics = np.unique([grp[0] for grp in group_names])
    tract_names = np.unique([grp[1] for grp in group_names])

    width, height = set_size(width=TEXT_WIDTH, subplots=(1, 3))
    fig, axes = plt.subplots(1, 3, figsize=(width, height * 1.3))

    group_by = dataset.y[:, 0]
    group_by_name = "QC"

    if len(group_by.shape) == 1:
        group_by = np.copy(group_by)[:, np.newaxis]

    groups_metric = [grp for gn, grp in zip(group_names, groups) if metrics[0] in gn]
    palette = "plasma"
    bins = 4
    ci = 95.0

    for metric, ax in zip(tqdm(metrics), axes[:2]):
        X_metric = GroupExtractor(
            select=metric, groups=groups, group_names=group_names
        ).fit_transform(X_harmonized)
        group_names_metric = [gn for gn in group_names if metric in gn]

        # Create a dataframe for each bundle
        tract_stats = {}
        for tid in tract_names:
            X_select = GroupExtractor(
                select=tid, groups=groups_metric, group_names=group_names_metric
            ).fit_transform(X_metric)
            columns = [idx for idx in range(X_select.shape[1])]
            df = pd.concat(
                [
                    pd.DataFrame(X_select, columns=columns, dtype=np.float64),
                    pd.DataFrame(group_by, columns=["group_by"]),
                ],
                axis="columns",
            )
            id_vars = ["group_by"]
            hue = "group_by"  # used later in seaborn functions

            df["bin"], _bins = pd.cut(
                df["group_by"].astype(np.float64), bins, retbins=True
            )
            hue = "bin"
            id_vars.append("bin")

            df = df.melt(id_vars=id_vars, var_name="nodeID", value_name=metric)
            df["nodeID"] = df["nodeID"].astype(int)
            tract_stats[tid] = df

        # Arrange the bundles into a grid
        bgcolor = "white"

        for tid, df_stat in tqdm(tract_stats.items()):
            if metric == "dki_md":
                df_stat[metric] *= 1000.0

            _ = sns.lineplot(
                x="nodeID",
                y=metric,
                hue=hue,
                data=df_stat,
                ci=ci,
                palette=palette,
                ax=ax,
                linewidth=1.0,
                n_boot=500,
            )

            _ = ax.set_xlabel("%% distance along fiber bundle")
            _ = ax.set_ylabel(metric.lower().replace("_", " "), labelpad=1)
            _ = ax.tick_params(axis="both", which="major")
            _ = ax.set_facecolor(bgcolor)
            _ = ax.get_legend().remove()

    handles, labels = axes[1].get_legend_handles_labels()

    labels = [f"QC {b[0]:.2f}-{b[1]:.2f}" for b in zip(_bins[:-1], _bins[1:])]

    return fig, axes, handles, labels


def age_predict_qc_sweep(out_dir, csv_dir, recompute=False, model_type="xgb"):
    n_splits = 5
    if recompute or not op.exists(op.join(csv_dir, "age_predict_qc_sweep.csv")):
        tsv_path = "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv"
        nodes_path = (
            "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/afq/combined_tract_profiles.csv",
        )

        dataset = AFQDataset.from_files(
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

    fig, axes, bundle_handles, bundle_labels = plot_slfl_profiles()

    qc_df["Training set size"] = (n_splits - 1) / n_splits * qc_df["train_size"]
    colors = plt.get_cmap("tab10").colors

    ax0 = axes[2]

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
        lw=1.25,
        ax=ax1,
        color=colors[2],
        label="Training set size (right axis)",
    )
    handles1, _ = ax1.get_legend_handles_labels()
    ax1.get_legend().remove()
    handles0, _ = ax0.get_legend_handles_labels()
    handles = handles0 + handles1
    labels = [r"$R^2$", "Training size"]

    _ = ax0.set_xlabel("QC cutoff")
    _ = ax1.set_xlabel("QC cutoff")
    _ = ax0.set_ylabel(r"Age prediction $R^2$", labelpad=1)

    handles = bundle_handles + handles
    labels = bundle_labels + labels

    figlegend_kwargs = dict(
        facecolor="whitesmoke",
        bbox_to_anchor=(0.5, 0.08),
        loc="upper center",
        ncol=6,
        handlelength=1.5,
        handletextpad=0.5,
    )

    leg = plt.figlegend(
        handles,
        labels,
        **figlegend_kwargs,
    )

    for legobj in leg.legendHandles:
        _ = legobj.set_linewidth(2.0)

    for letter, ax in zip("abc", axes.flatten()):
        these_kwargs = TEXT_KWARGS.copy()
        ax.text(
            s=letter,
            transform=ax.transAxes,
            **these_kwargs,
        )

    fig_tight_layout_kws = dict(w_pad=0.1)
    fig.tight_layout(**fig_tight_layout_kws)

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
