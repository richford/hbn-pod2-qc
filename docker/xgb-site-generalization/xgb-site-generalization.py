#!/opt/conda/bin/python

import argparse
import dask.dataframe as dd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import pandas as pd
import re
import s3fs
import seaborn as sns

from plot_formatting import set_size, FULL_WIDTH, TEXT_WIDTH
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV
from tqdm.auto import tqdm, trange
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

SWARM_SIZE = 5

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


def _get_Xy(expert_rating_file, fibr_dir, image_type=None):
    # The classification targets are the expert ratings, scaled from zero to one
    # and then binarized
    y = pd.read_csv(expert_rating_file, index_col="subject")
    scaler = MinMaxScaler()
    y_prob = pd.Series(
        data=np.squeeze(scaler.fit_transform(y[["rating"]])), index=y.index
    )
    y_binary = (y_prob >= 0.5).astype(int)

    # The classification features are a combination of the mean fibr rating from
    # each user and the automated QC metrics from qsiprep.
    # First, load the automated QC metrics, dropping the non-numerical columns
    fs = s3fs.S3FileSystem(anon=True)
    dwiqc_s3_uri = (
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/dwiqc.json"
    )
    with fs.open(dwiqc_s3_uri) as fp:
        qc_json = json.load(fp)

    df_qc = pd.DataFrame(qc_json["subjects"])
    df_qc["subject_id"] = df_qc["subject_id"].apply(lambda s: s.replace("sub-", ""))
    df_qc.set_index("subject_id", drop=True, inplace=True)
    df_qc.to_csv(op.join(fibr_dir, "dwiqc.csv"))
    df_qc.drop(
        [
            "participant_id",
            "task_id",
            "dir_id",
            "acq_id",
            "file_name",
            "run_id",
            "space_id",
            "rec_id",
        ],
        axis="columns",
        inplace=True,
    )

    # Now load the fibr ratings
    with open(op.join(fibr_dir, "manifest.json")) as fp:
        manifest = json.load(fp)

    df_votes = pd.read_csv(op.join(fibr_dir, "votes.csv"))
    r = re.compile("sub-([a-zA-Z0-9]*)_")
    df_votes["subject"] = df_votes["sample"].apply(
        lambda s: r.search(s).group(1) if r.search(s) is not None else np.na
    )
    df_votes = df_votes[df_votes["sample"].isin(manifest)]

    # Keep only fibr raters with over 100 ratings
    users = df_votes[["user", "response"]].groupby("user").count()
    valid_users = users[users["response"] >= 100].index.to_list()
    df_valid_votes = df_votes[df_votes["user"].isin(valid_users)].copy()
    del df_votes

    if image_type is not None:
        df_valid_votes["image_type"] = [
            "animated" if not s.endswith("_5") else "static"
            for s in df_valid_votes["sample"]
        ]

        if image_type in ["animated", "static"]:
            df_valid_votes = df_valid_votes[df_valid_votes["image_type"] == image_type]
        elif image_type == "both":
            df_valid_votes["user"] = (
                df_valid_votes["user"] + "_" + df_valid_votes["image_type"]
            )
        else:
            raise ValueError(
                f"image_type must be 'animated', 'static', 'both', or None. Got {image_type} instead."
            )

        df_valid_votes.drop("image_type", axis="columns", inplace=True)

    fibr_votes = (
        df_valid_votes.groupby(["subject", "user"], as_index=False)["response"]
        .mean()
        .pivot(
            index="subject",
            columns="user",
            values="response",
        )
    )

    # Create the feature matrix by merging the fibr ratings with the qc ratings
    # Keep merging "left" to keep only the subjects with fibr ratings.
    X = (
        pd.DataFrame(index=y.index)
        .merge(fibr_votes, how="left", left_index=True, right_index=True)
        .merge(
            df_qc,
            how="left",
            left_index=True,
            right_index=True,
        )
    )

    return X, y_binary, df_qc, fibr_votes


def xgb_site_generalization(
    expert_rating_file,
    fibr_dir,
    xgb_model_dir,
):
    # Create the xgb model directory if it doesn't already exist
    os.makedirs(op.abspath(xgb_model_dir), exist_ok=True)

    _, y, df_qc, _ = _get_Xy(
        expert_rating_file=expert_rating_file, fibr_dir=fibr_dir, image_type=None
    )

    df_qc["site"] = df_qc["session_id"].apply(lambda s: s.split("site")[-1])
    df_qc.drop(columns=["session_id"], inplace=True)

    df_qc_gold = pd.DataFrame(index=y.index).merge(
        df_qc,
        how="left",
        left_index=True,
        right_index=True,
    )

    params = {
        "n_estimators": (50, 5001),
        "min_child_weight": (1, 11),
        "gamma": (0.001, 5.0, "log-uniform"),
        "eta": (0.001, 0.5, "log-uniform"),
        "subsample": (0.2, 1.0),
        "colsample_bytree": (0.1, 1.0),
        "max_depth": (2, 8),
    }

    site_experiments = [
        {"train": ["RU", "CUNY"], "test": ["CBIC"]},
        {"train": ["CBIC"], "test": ["RU", "CUNY"]},
        {"train": ["CBIC", "CUNY"], "test": ["RU"]},
        {"train": ["RU"], "test": ["CBIC", "CUNY"]},
    ]

    models = {}
    checkpointed = {}
    results_dicts = []
    predictions_dfs = []

    for site_experiment in tqdm(site_experiments, leave=False, ncols=80, position=0):
        train_sites = site_experiment["train"]
        test_sites = site_experiment["test"]

        df_train = (
            df_qc_gold[df_qc_gold["site"].isin(train_sites)]
            .drop(columns=["site"])
            .to_numpy()
        )

        df_test = (
            df_qc_gold[df_qc_gold["site"].isin(test_sites)]
            .drop(columns=["site"])
            .to_numpy()
        )

        y_train = y[df_qc_gold["site"].isin(train_sites)].copy()
        y_test = y[df_qc_gold["site"].isin(test_sites)].copy()

        site_string = "train_" + "_".join(train_sites) + "_test_" + "_".join(test_sites)
        models[site_string] = {}
        checkpointed[site_string] = {}

        for random_state in trange(20, leave=False, ncols=80, position=1):
            model_json = op.join(
                xgb_model_dir, f"{site_string}_seed-{random_state}.json"
            )

            try:
                xgb = XGBClassifier()
                xgb.load_model(model_json)
                checkpointed[site_string][random_state] = True
                models[site_string][random_state] = xgb
            except XGBoostError:
                cv = RepeatedStratifiedKFold(
                    n_splits=3, n_repeats=2, random_state=random_state
                )

                models[site_string][random_state] = BayesSearchCV(
                    XGBClassifier(
                        nthread=4, use_label_encoder=False, eval_metric="logloss"
                    ),
                    params,
                    n_iter=100,
                    scoring="neg_log_loss",
                    verbose=0,
                    return_train_score=True,
                    cv=cv,
                    random_state=random_state,
                )

                models[site_string][random_state].fit(df_train, y_train)
                models[site_string][random_state].best_estimator_.save_model(model_json)

            y_hat = models[site_string][random_state].predict(df_test)
            y_prob = models[site_string][random_state].predict_proba(df_test)[:, 1]
            pretty_site_string = (
                "train: " + "+".join(train_sites) + ", test: " + "+".join(test_sites)
            )

            df_prediction = pd.DataFrame()
            df_prediction["y_true"] = y_test
            df_prediction["sites"] = pretty_site_string
            df_prediction["y_prob"] = y_prob
            df_prediction["seed"] = random_state
            df_prediction.reset_index(inplace=True)
            predictions_dfs.append(df_prediction)

            for _score, _metric, _y in zip(
                [accuracy_score, balanced_accuracy_score, roc_auc_score],
                ["Accuracy", "Balanced accuracy", "ROC AUC"],
                [y_hat, y_hat, y_prob],
            ):
                results_dicts.append(
                    {
                        "site": pretty_site_string,
                        "seed": random_state,
                        "score": _score(y_test, _y),
                        "metric": _metric,
                    }
                )

    df_performance = pd.DataFrame(results_dicts)
    df_predictions = pd.concat(predictions_dfs)

    return df_predictions, df_performance


def plot_dl_site_generalization(dl_prediction_dir, fig_dir):
    participants = pd.read_csv(
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        sep="\t",
    )
    participants["XGB pass?"] = participants["xgb_qc_score"] >= 0.5
    participants["Expert pass?"] = participants["expert_qc_score"] >= 0.5
    participants.rename(
        columns={
            "xgb_qc_score": "XGB rating",
            "expert_qc_score": "Expert rating",
            "dl_qc_score": "CNN-i rating",
            "scan_site_id": "Site",
        },
        inplace=True,
    )

    report_glob = op.join(dl_prediction_dir, "*_report_set_*.csv")
    test_glob = op.join(dl_prediction_dir, "*_test_set_*.csv")

    def get_report_df(glob):
        df = dd.read_csv(glob).rename(
            columns={"Unnamed: 0": "subject_id", "y_from_tsv": "y_target"}
        )

        df = df.compute().reset_index(drop=True)
        merged = pd.merge(df, participants, how="left", on="subject_id")
        merged.drop(columns=["ehq_total", "commercial_use", "full_pheno"], inplace=True)
        return merged

    df_prediction = {
        "test": get_report_df(test_glob),
        "report": get_report_df(report_glob),
    }

    df_prediction["test"]["sites"] = (
        "train: "
        + df_prediction["test"]["train_sites"]
        + ", test: "
        + df_prediction["test"]["test_sites"]
    )
    df_prediction["report"]["sites"] = (
        "train: "
        + df_prediction["report"]["train_sites"]
        + ", test: "
        + df_prediction["report"]["test_sites"]
    )

    sites = sorted(pd.unique(df_prediction["test"]["sites"]))

    df_performance = {}

    for split, df in df_prediction.items():
        dataframe_dicts = []
        for site in sites:
            df_site = df[df["sites"] == site]
            for seed in pd.unique(df_site["seed"]):
                df_seed = df_site[df_site["seed"] == seed]

                dataframe_dicts.append(
                    dict(
                        site=site,
                        seed=seed,
                        score=roc_auc_score(df_seed["y_true"], df_seed["y_prob"]),
                        metric="ROC AUC",
                    )
                )
                dataframe_dicts.append(
                    dict(
                        site=site,
                        seed=seed,
                        score=accuracy_score(
                            df_seed["y_true"], df_seed["y_prob"] >= 0.5
                        ),
                        metric="Accuracy",
                    )
                )
                dataframe_dicts.append(
                    dict(
                        site=site,
                        seed=seed,
                        score=balanced_accuracy_score(
                            df_seed["y_true"], df_seed["y_prob"] >= 0.5
                        ),
                        metric="Balanced accuracy",
                    )
                )

        df_performance[split] = pd.DataFrame(dataframe_dicts)

    colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=set_size(width=FULL_WIDTH, subplots=(3, 2)))

    spec = fig.add_gridspec(3, 2, hspace=0.45)
    ax0 = fig.add_subplot(spec[0, :])
    ax1 = fig.add_subplot(spec[1, :])
    ax2 = fig.add_subplot(spec[2, 0])
    ax3 = fig.add_subplot(spec[2, 1])
    ax2.get_shared_y_axes().join(ax2, ax3)

    for (split, df), ax in zip(df_performance.items(), [ax0, ax1]):
        _ = sns.stripplot(
            data=df,
            x="site",
            y="score",
            hue="metric",
            order=sites,
            hue_order=["ROC AUC", "Accuracy", "Balanced accuracy"],
            dodge=True,
            ax=ax,
            size=SWARM_SIZE,
            palette=colors,
        )

        xlabels = ax.get_xticklabels()
        xlabels = [label.get_text().replace(", ", "\n") for label in xlabels]
        ax.set_xticklabels(xlabels)
        ax.set_title(split.capitalize() + " set")

    axes = [ax2, ax3]
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for ax, (split, df_split) in zip(axes, df_prediction.items()):
        for site, color, marker in zip(sites, colors[3:], ["s", "^", "X", "*"]):
            df = df_split[df_split["sites"] == site].copy()

            if split == "report":
                y = "Expert rating"
            else:
                y = "XGB rating"

            prob_true, prob_pred = calibration_curve(
                y_true=(df[y] > 0.5).astype(int),
                y_prob=df["y_prob"],
                n_bins=5,
            )

            ax.plot(
                prob_pred, prob_true, ls="-", marker=marker, color=color, label=site
            )

        ax.set_title(f"{split.capitalize()} Calibration curve (reliability diagram)")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax0.get_legend().remove()
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(1.1, -0.2),
        framealpha=1.0,
        ncol=3,
        borderpad=0.2,
        columnspacing=1.0,
        handletextpad=0.2,
    )

    fig.savefig(op.join(fig_dir, "dl_site_generalization.pdf"), bbox_inches="tight")


def plot_xgb_site_generalization(
    expert_rating_file,
    fibr_dir,
    xgb_model_dir,
    fig_dir,
):
    df_prediction, df_performance = xgb_site_generalization(
        expert_rating_file=expert_rating_file,
        fibr_dir=fibr_dir,
        xgb_model_dir=xgb_model_dir,
    )

    colors = plt.get_cmap("tab10").colors
    sites = sorted(pd.unique(df_prediction["sites"]))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=set_size(width=FULL_WIDTH, subplots=(1, 2)),
        gridspec_kw=dict(width_ratios=[0.6, 0.4]),
    )
    fig.tight_layout()

    ax0 = axes[0]
    ax1 = axes[1]

    _ = sns.stripplot(
        data=df_performance,
        x="site",
        y="score",
        hue="metric",
        order=sites,
        hue_order=["ROC AUC", "Accuracy", "Balanced accuracy"],
        dodge=True,
        ax=ax0,
        size=SWARM_SIZE,
        palette=colors,
    )

    xlabels = ax0.get_xticklabels()
    xlabels = [label.get_text().replace(", ", "\n") for label in xlabels]
    ax0.set_xticklabels(xlabels)

    handles, labels = ax0.get_legend_handles_labels()
    ax0.get_legend().remove()

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for site, color, marker in zip(sites, colors[3:], ["s", "^", "X", "*"]):
        df = df_prediction[df_prediction["sites"] == site].copy()

        prob_true, prob_pred = calibration_curve(
            y_true=(df["y_true"] > 0.5).astype(int),
            y_prob=df["y_prob"],
            n_bins=5,
        )

        ax1.plot(prob_pred, prob_true, marker=marker, ls="-", color=color, label=site)

    ax1.set_title("Calibration curve (reliability diagram)")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = handles + handles1
    labels = labels + labels1

    transposed_indices = [0, 4, 1, 5, 2, 6, 3, 7]
    handles = [handles[i] for i in transposed_indices]
    labels = [labels[i] for i in transposed_indices]

    ax0.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.1, -0.25),
        framealpha=1.0,
        ncol=4,
        borderaxespad=0.0,
        borderpad=0.2,
        columnspacing=1.0,
        handletextpad=0.2,
    )

    fig.savefig(op.join(fig_dir, "xgb_site_generalization.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dl_prediction_dir",
        help="Directory containing the deep learning site generalization predictions.",
    )
    parser.add_argument(
        "expert_rating_file",
        help="CSV file containing average expert QC ratings.",
    )
    parser.add_argument(
        "fibr_dir",
        help="Directory containing Fibr data.",
    )
    parser.add_argument(
        "xgb_model_dir",
        help="Directory in which to save/load XGBoost models.",
    )
    parser.add_argument(
        "fig_dir",
        help="Figure directory.",
    )

    args = parser.parse_args()

    with plt.style.context("/tex.mplstyle"):
        plot_dl_site_generalization(
            dl_prediction_dir=args.dl_prediction_dir,
            fig_dir=args.fig_dir,
        )
        plot_xgb_site_generalization(
            expert_rating_file=args.expert_rating_file,
            fibr_dir=args.fibr_dir,
            xgb_model_dir=args.xgb_model_dir,
            fig_dir=args.fig_dir,
        )
