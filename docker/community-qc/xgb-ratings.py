#!/opt/conda/bin/python

import argparse
import dask.dataframe as dd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import pandas as pd
import pingouin as pg
import re
import s3fs
import seaborn as sns
import shap

from glob import glob
from plot_formatting import set_size, FULL_WIDTH, TEXT_WIDTH
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from skopt import BayesSearchCV
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

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
            "session_id",
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


def _save_participants_tsv(qc_ratings, output_dir, image_type=None):
    tsv_filename = "participants"
    if image_type is not None:
        tsv_filename += "_" + image_type

    if not op.exists(op.join(output_dir, f"{tsv_filename}.tsv")):
        # Save these qc labels in a participants.tsv file
        fs = s3fs.S3FileSystem(anon=True)
        dwiqc_s3_uri = "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/dwiqc.json"
        with fs.open(dwiqc_s3_uri) as fp:
            qc_json = json.load(fp)

        dwiqc = pd.DataFrame(qc_json["subjects"])[["subject_id", "session_id"]]
        dwiqc["EID"] = dwiqc["subject_id"].apply(lambda s: s.replace("sub-", ""))
        dwiqc["scan_site_id"] = dwiqc["session_id"].apply(
            lambda s: s.replace("ses-HBNsite", "")
        )
        dwiqc.drop("session_id", axis="columns", inplace=True)
        dwiqc.set_index("EID", inplace=True)
        dwiqc = dwiqc.merge(qc_ratings, left_index=True, right_index=True)

        html_files = fs.glob(
            "fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/sub-*.html"
        )
        html_paths = [sub.replace(".html", "") for sub in html_files]
        s3_paths = [fs.glob(sub + "/ses-HBNsite*")[0] for sub in html_paths]
        s3_subs = [path.split("/qsiprep/")[1].split("/")[0] for path in s3_paths]
        s3_sites = [path.split("ses-HBNsite")[-1] for path in s3_paths]
        df_s3 = pd.DataFrame(index=s3_subs)
        df_s3["scan_site_id"] = s3_sites
        df_s3.index.name = "subject_id"

        df_participants = df_s3.merge(
            dwiqc.drop("scan_site_id", axis="columns"),
            how="left",
            left_index=True,
            right_on="subject_id",
        ).set_index("subject_id", drop=True)

        df_participants.to_csv(
            op.join(output_dir, f"{tsv_filename}.tsv"), sep="\t", na_rep="n/a"
        )


def xgb_qc(
    expert_rating_file,
    fibr_dir,
    xgb_model_dir,
    output_dir,
    fig_dir,
    random_state=42,
    image_type=None,
):
    # Create the xgb model directory if it doesn't already exist
    os.makedirs(op.abspath(xgb_model_dir), exist_ok=True)

    X, y, df_qc, fibr_votes = _get_Xy(
        expert_rating_file=expert_rating_file, fibr_dir=fibr_dir, image_type=image_type
    )

    Xqc = (
        pd.DataFrame(index=y.index)
        .merge(
            df_qc,
            how="left",
            left_index=True,
            right_index=True,
        )
        .to_numpy()
    )

    Xfibr = (
        pd.DataFrame(index=y.index)
        .merge(fibr_votes, how="left", left_index=True, right_index=True)
        .to_numpy()
    )

    X_columns = X.columns
    y = y.to_numpy()
    X = X.to_numpy()

    # Create parameter distributions for XGBoost
    params = {
        "n_estimators": (50, 5001),
        "min_child_weight": (1, 11),
        "gamma": (0.001, 5.0, "log-uniform"),
        "eta": (0.001, 0.5, "log-uniform"),
        "subsample": (0.2, 1.0),
        "colsample_bytree": (0.1, 1.0),
        "max_depth": (2, 8),
    }

    # Train different XGBoost classifiers for each CV split
    random_state = random_state
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=random_state)

    checkpointed = {}
    qc_checkpointed = {}
    shap_values = []
    f_values = []
    qc_shap_values = []
    qc_f_values = []
    mean_fpr = np.linspace(0, 1, 100)

    models, qc_models, fibr_models = {}, {}, {}
    tprs, qc_tprs, fibr_tprs = [], [], []
    aucs, qc_aucs, fibr_aucs = [], [], []

    model_name = "xgb-classifier"
    qc_model_name = "xgb-qc-classifier"
    fibr_model_name = "xgb-fibr-classifier"

    if image_type is not None:
        model_name += "_" + image_type
        qc_model_name += "_" + image_type
        fibr_model_name += "_" + image_type

    # While training, plot the roc curve for each split
    # Thanks sklearn examples, you rock!
    fig, ax = plt.subplots(1, 1, figsize=set_size(width=0.37 * FULL_WIDTH))
    for i, (train, test) in enumerate(tqdm(cv.split(X, y), total=6)):
        try:
            xgb = XGBClassifier()
            xgb.load_model(
                op.join(xgb_model_dir, f"{model_name}_seed-{random_state}_cv-{i}.json")
            )
            checkpointed[i] = True
            models[i] = xgb
            explainer = shap.Explainer(xgb, feature_names=X_columns)
            shap_values.append(explainer(X[test]))
            f_values.append(xgb.feature_importances_)
        except XGBoostError:
            models[i] = BayesSearchCV(
                XGBClassifier(
                    nthread=4, use_label_encoder=False, eval_metric="logloss"
                ),
                params,
                n_iter=100,
                scoring="neg_log_loss",
                verbose=1,
            )
            models[i].fit(X[train], y[train])
            models[i].best_estimator_.save_model(
                op.join(xgb_model_dir, f"{model_name}_seed-{random_state}_cv-{i}.json")
            )
            checkpointed[i] = False
            explainer = shap.Explainer(
                models[i].best_estimator_, feature_names=X_columns
            )
            shap_values.append(explainer(X[test]))
            f_values.append(models[i].best_estimator_.feature_importances_)

        try:
            xgb = XGBClassifier()
            xgb.load_model(
                op.join(
                    xgb_model_dir, f"{qc_model_name}_seed-{random_state}_cv-{i}.json"
                )
            )
            qc_checkpointed[i] = True
            qc_models[i] = xgb
            explainer = shap.Explainer(xgb, feature_names=df_qc.columns)
            qc_shap_values.append(explainer(Xqc[test]))
            qc_f_values.append(xgb.feature_importances_)
        except XGBoostError:
            qc_models[i] = BayesSearchCV(
                XGBClassifier(
                    nthread=4, use_label_encoder=False, eval_metric="logloss"
                ),
                params,
                n_iter=100,
                scoring="neg_log_loss",
                verbose=1,
            )
            qc_models[i].fit(Xqc[train], y[train])
            qc_models[i].best_estimator_.save_model(
                op.join(
                    xgb_model_dir, f"{qc_model_name}_seed-{random_state}_cv-{i}.json"
                )
            )
            qc_checkpointed[i] = False
            explainer = shap.Explainer(
                qc_models[i].best_estimator_, feature_names=df_qc.columns
            )
            qc_shap_values.append(explainer(Xqc[test]))
            qc_f_values.append(models[i].best_estimator_.feature_importances_)

        try:
            xgb = XGBClassifier()
            xgb.load_model(
                op.join(
                    xgb_model_dir, f"{fibr_model_name}_seed-{random_state}_cv-{i}.json"
                )
            )
            fibr_models[i] = xgb
        except XGBoostError:
            fibr_models[i] = BayesSearchCV(
                XGBClassifier(
                    nthread=4, use_label_encoder=False, eval_metric="logloss"
                ),
                params,
                n_iter=100,
                scoring="neg_log_loss",
                verbose=1,
            )
            fibr_models[i].fit(Xfibr[train], y[train])
            fibr_models[i].best_estimator_.save_model(
                op.join(
                    xgb_model_dir, f"{fibr_model_name}_seed-{random_state}_cv-{i}.json"
                )
            )

        for _model, _X, _tprs, _aucs in zip(
            [models[i], qc_models[i], fibr_models[i]],
            [X, Xqc, Xfibr],
            [tprs, qc_tprs, fibr_tprs],
            [aucs, qc_aucs, fibr_aucs],
        ):
            fpr, tpr, _ = roc_curve(y[test], _model.predict_proba(_X[test])[:, 1])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            _tprs.append(interp_tpr)
            _aucs.append(roc_auc_score(y[test], _model.predict_proba(_X[test])[:, 1]))

    colors = plt.get_cmap("tab10").colors

    for _tprs, _aucs, color, label in zip(
        [tprs, qc_tprs, fibr_tprs],
        [aucs, qc_aucs, fibr_aucs],
        colors,
        ["XGB", "XGB-q", "XGB-f"],
    ):
        mean_tpr = np.mean(_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(_aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            label=r"%s (%0.2f $\pm$ %0.2f)" % (label, mean_auc, std_auc),
            lw=1.5,
            alpha=0.8,
        )

        std_tpr = np.std(_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev." if label == "Fibr + dwiqc" else None,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=1.5,
        color=colors[3],
        label="Chance",
        alpha=0.8,
    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.01, -0.025),
        framealpha=1.0,
        title="Model (AUC)",
        labelspacing=0.25,
        handlelength=1.25,
        handletextpad=0.5,
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    these_kwargs = TEXT_KWARGS.copy()
    these_kwargs.update(dict(x=-0.175))
    ax.text(
        s="c",
        transform=ax.transAxes,
        **these_kwargs,
    )

    plot_title = "xgb-roc-curve"
    shap_csv_title = "xgb_shap_values"
    shap_q_csv_title = "xgb_q_shap_values"
    qc_csv_title = "qc_ratings"
    if image_type is not None:
        plot_title += "-" + image_type
        shap_csv_title += "_" + image_type
        shap_q_csv_title += "_" + image_type
        qc_csv_title += "_" + image_type

    fig.savefig(op.join(fig_dir, f"{plot_title}.pdf"), bbox_inches="tight")

    absmean_shap = pd.Series(
        index=X_columns,
        data=np.absolute(np.concatenate([sh.values for sh in shap_values])).mean(
            axis=0
        ),
        name="mean-absolute-shap-value",
    )
    absmean_shap = absmean_shap.loc[absmean_shap.index.isin(df_qc.columns)]
    absmean_shap.index.rename("feature", inplace=True)
    absmean_shap.sort_values(ascending=False, inplace=True)
    absmean_shap.to_csv(op.join(output_dir, f"{shap_csv_title}.csv"))
    absmean_shap.to_latex(op.join(output_dir, f"{shap_csv_title}.tex"), longtable=False)

    absmean_shap = pd.Series(
        index=df_qc.columns,
        data=np.absolute(np.concatenate([sh.values for sh in qc_shap_values])).mean(
            axis=0
        ),
        name="mean-absolute-shap-value",
    )
    absmean_shap.index.rename("feature", inplace=True)
    absmean_shap.sort_values(ascending=False, inplace=True)
    absmean_shap.to_csv(op.join(output_dir, f"{shap_q_csv_title}.csv"))
    absmean_shap.to_latex(
        op.join(output_dir, f"{shap_q_csv_title}.tex"), longtable=False
    )

    # Create a voting classifier from each CV's XGBoost classifier.
    # Weight each classifier by its out-of-sample ROC AUC
    estimators = {
        f"cv{i}": models[i] if checkpointed[i] else models[i].best_estimator_
        for i in range(6)
    }

    qc_estimators = {
        f"cv{i}": qc_models[i] if qc_checkpointed[i] else qc_models[i].best_estimator_
        for i in range(6)
    }

    weights = aucs
    qc_weights = qc_aucs
    voter = VotingClassifier(estimators=estimators, weights=weights, voting="soft")
    qc_voter = VotingClassifier(
        estimators=qc_estimators, weights=qc_weights, voting="soft"
    )

    # We don't want to refit so manually set the fitted estimators
    voter.estimators_ = list(estimators.values())
    voter.le_ = LabelEncoder().fit(y)
    voter.classes_ = voter.le_.classes_

    qc_voter.estimators_ = list(qc_estimators.values())
    qc_voter.le_ = LabelEncoder().fit(y)
    qc_voter.classes_ = qc_voter.le_.classes_

    fig, ax = plt.subplots(1, 1, figsize=set_size(width=0.5 * FULL_WIDTH))

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf, label, _X in zip([voter, qc_voter], ["XGB", "XGB-q"], [X, Xqc]):
        prob_true, prob_pred = calibration_curve(
            y_true=y,
            y_prob=clf.predict_proba(_X)[:, 1],
            n_bins=5,
        )
        ax.plot(prob_pred, prob_true, "s-", label=label)

    ax.set_title("Calibration curve (reliability diagram)")
    ax.legend(loc="upper left", framealpha=1.0, title="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")

    fig.savefig(op.join(fig_dir, "xgb-calibration-curve.pdf"), bbox_inches="tight")

    # Now create qc ratings for all subjects, not just the ones in the gold
    # standard dataset.
    X_all_subjects = fibr_votes.merge(
        df_qc,
        how="inner",
        left_index=True,
        right_index=True,
    )
    y_all_subjects = voter.predict_proba(X_all_subjects)[:, 1]
    qc_ratings = pd.DataFrame(index=X_all_subjects.index)
    qc_ratings["fibr + qsiprep rating"] = y_all_subjects

    df_qc.to_csv(op.join(output_dir, "dwiqc.csv"))
    qsiprep_all_subjects = qc_voter.predict_proba(df_qc)[:, 1]
    qsiprep_ratings = pd.DataFrame(index=df_qc.index)
    qsiprep_ratings["qsiprep rating"] = qsiprep_all_subjects

    qc_ratings = qsiprep_ratings.merge(
        qc_ratings, how="left", left_index=True, right_index=True
    )
    qc_ratings.to_csv(op.join(output_dir, f"{qc_csv_title}.csv"))
    # print(qc_weights)

    # _save_participants_tsv(qc_ratings, output_dir, image_type=image_type)


def compute_irr_with_xgb_rater(expert_qc_dir, fibr_deriv_dir, fig_dir):
    qc_files = glob(op.join(expert_qc_dir, "rater-*_qc.csv"))

    expert_qc = dd.read_csv(
        qc_files, usecols=["subject", "rating"], include_path_column=True
    ).compute()
    expert_qc["rater"] = (
        expert_qc["path"]
        .apply(lambda s: op.basename(s).replace("rater-", "").replace("_qc.csv", ""))
        .astype(str)
    )
    expert_qc.drop("path", axis="columns", inplace=True)
    expert_qc.reset_index(drop=True, inplace=True)

    ratings = expert_qc.pivot(index="subject", columns="rater", values="rating")

    fibr = pd.read_csv(op.join(fibr_deriv_dir, "participants.tsv"), sep="\t")
    fibr["subject_id"] = fibr["subject_id"] + "_ses-HBNsite" + fibr["scan_site_id"]
    fibr.drop(columns=["scan_site_id", "qsiprep rating"], inplace=True)
    fibr.dropna(inplace=True)
    fibr.columns = ["subject", "rating"]
    fibr["rating"] = np.rint((fibr["rating"] * 4 - 2).to_numpy()).astype(int)
    fibr["rater"] = "xgb"
    fibr = fibr[fibr["subject"].isin(ratings.index.to_list())]

    expert_qc_with_xgb = pd.concat([expert_qc, fibr])

    icc_with_xgb = (
        pg.intraclass_corr(
            data=expert_qc_with_xgb, targets="subject", raters="rater", ratings="rating"
        )
        .round(3)
        .set_index("Type")
        .filter(like="ICC3", axis="index")
    )
    icc_with_xgb.to_csv(op.join(fibr_deriv_dir, "icc.csv"))

    fibr.set_index("subject", inplace=True)
    fibr.drop(columns="rater", inplace=True)
    fibr.columns = pd.Index(["xgb"], name="rater")

    ratings = pd.merge(ratings, fibr, left_index=True, right_index=True)

    cohen_kappas = {}
    for rater in ratings.columns:
        cohen_kappas[rater] = [
            cohen_kappa_score(
                ratings[rater].to_numpy(),
                ratings[other_rater].to_numpy(),
                labels=[-2, -1, 0, 1, 2],
                weights="quadratic",
            )
            for other_rater in ratings.columns
        ]

    df_kappa = pd.DataFrame(cohen_kappas, index=ratings.columns)
    for rater in ratings.columns:
        df_kappa.loc[rater, rater] = np.nan

    df_kappa["mean"] = df_kappa.mean(axis="columns")

    # plot the heatmap for correlation matrix
    grid_kws = {"width_ratios": (0.9, 0.075), "wspace": 0.15}
    width, height = set_size(width=0.4 * FULL_WIDTH)
    fig, (ax, cbar_ax) = plt.subplots(
        1, 2, gridspec_kw=grid_kws, figsize=(width, height * 1.5)
    )

    vmin = df_kappa.to_numpy().min()
    vmax = df_kappa.to_numpy().max()

    mapper = {
        "awedTortoise8": "A",
        "eagerHawk8": "B",
        "gloomyJerky1": "C",
        "grudgingBass4": "D",
        "wornoutRhino9": "E",
        "wrathfulMuesli7": "F",
    }
    df_kappa = df_kappa.rename(columns=mapper).rename(index=mapper)

    ax = sns.heatmap(
        df_kappa,
        # vmin=vmin, vmax=vmax, center=0.5,
        cmap=sns.color_palette("Blues", as_cmap=True),
        ax=ax,
        cbar_ax=cbar_ax,
        # square=False,
        annot=True,
        fmt=".2f",
        mask=np.tril(np.ones_like(df_kappa.to_numpy())),
    )

    _ = ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _ = ax.set_xticklabels(ax.get_xticklabels())
    _ = ax.set_ylabel("rater", labelpad=-1)
    _ = ax.set_xlabel("rater")
    these_kwargs = TEXT_KWARGS.copy()
    these_kwargs.update(dict(x=-0.05, y=1.05))
    _ = ax.text(
        s="e",
        transform=ax.transAxes,
        **these_kwargs,
    )
    ax.set_title(r"Pairwise Cohen's $\kappa$")
    fig.savefig(op.join(fig_dir, "expert-raters-cohens-kappa.pdf"), bbox_inches="tight")

    df_kappa.loc["mean"] = df_kappa["mean"].to_list() + [df_kappa.mean().mean()]
    df_kappa.to_csv(op.join(fibr_deriv_dir, "cohens_kappa.csv"))


def plot_xgb_scatter(expert_rating_file, output_dir, fibr_dir, fig_dir):
    X, _, df_qc, fibr_votes = _get_Xy(
        expert_rating_file=expert_rating_file, fibr_dir=fibr_dir
    )
    y = pd.read_csv(expert_rating_file, index_col="subject")
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
    df_xgb_qc = pd.read_csv(
        op.join(output_dir, "qc_ratings.csv"), index_col="subject_id"
    )
    merged = pd.merge(
        pd.merge(
            y[["rating"]], df_xgb_qc, how="left", left_index=True, right_index=True
        ),
        pd.DataFrame(
            X[fibr_votes.columns].mean(axis="columns"), columns=["fibr rating"]
        ),
        left_index=True,
        right_index=True,
    )

    del df_xgb_qc

    fig, axes = plt.subplots(
        1,
        2,
        figsize=set_size(width=0.63 * FULL_WIDTH, subplots=(1, 2)),
        sharey=True,
    )
    fig.tight_layout(pad=0)

    for x_var, ax in zip(["fibr rating", "fibr + qsiprep rating"], axes):
        _ = sns.scatterplot(data=merged, x=x_var, y="rating", s=18, ax=ax)

    axes[0].set_ylabel("Expert rating", labelpad=-0.25)
    axes[0].set_xlabel("Mean Fibr rating")
    axes[1].set_xlabel("XGB rating")

    for letter, ax in zip("ab", axes.flatten()):
        ax.tick_params(axis="both", which="major")
        these_kwargs = TEXT_KWARGS.copy()
        if letter == "b":
            these_kwargs.update(dict(x=0))
        else:
            these_kwargs.update(dict(x=-0.1))
        ax.text(
            s=letter,
            transform=ax.transAxes,
            **these_kwargs,
        )

    del merged
    fig.savefig(op.join(fig_dir, "fibr-rating-scatter-plot.pdf"), bbox_inches="tight")
    del fig, axes

    X_all = pd.merge(X, y, left_index=True, right_index=True)

    del X, y

    X_all.rename(
        columns={
            "rating": "Expert rating",
            "raw_neighbor_corr": "Neighboring DWI correlation",
            "raw_num_bad_slices": "Num outlier slices",
            "max_rel_translation": "Max rel. translation",
        },
        inplace=True,
    )

    fig1, axes1 = plt.subplots(
        2, 2, figsize=set_size(width=0.6 * FULL_WIDTH, subplots=(2, 2))
    )
    fig1.tight_layout()

    for x_var, ax in zip(
        ["Neighboring DWI correlation", "Num outlier slices", "Max rel. translation"],
        axes1.flatten()[1:],
    ):
        _ = sns.scatterplot(data=X_all, x=x_var, y="Expert rating", s=14, ax=ax)
        r, _ = pearsonr(X_all[x_var], X_all["Expert rating"])
        print(f"Pearson_R({x_var}, Expert rating) = {r:.3f}")

    _ = sns.histplot(data=X_all, x="Expert rating", ax=axes1[0, 0])

    # tick_labels = axes1[0, 0].get_yticklabels()
    these_kwargs = TEXT_KWARGS.copy()
    these_kwargs.update(dict(y=1.1))
    for letter, ax in zip("abcd", axes1.flatten()):
        label = ax.get_xlabel()
        ax.set_xlabel(label, labelpad=-0.5)
        label = ax.get_ylabel()
        ax.set_ylabel(label, labelpad=-0.5)
        ax.text(
            s=letter,
            transform=ax.transAxes,
            **TEXT_KWARGS,
        )

    fig1.savefig(
        op.join(fig_dir, "expert-qsiprep-pairplot.pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "raw_expert_dir",
        help="Directory containing raw expert ratings.",
    )
    parser.add_argument(
        "output_dir",
        help="Output directory.",
    )
    parser.add_argument(
        "fig_dir",
        help="Figure directory.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for cross-validation splitter.",
    )
    parser.add_argument(
        "--image_type",
        choices=["static", "animated", "both"],
        default=None,
        help=(
            "Type of fibr image on which to estimate fibr ratings. By default, "
            "static image and animated image scores are averaged together. Use "
            "'static' or 'animated' to limit fibr ratings to only static or "
            "animated image types respectively. Use 'both' to compute separate "
            "mean ratings for each image type."
        ),
    )

    args = parser.parse_args()

    with plt.style.context("/tex.mplstyle"):
        xgb_qc(
            expert_rating_file=args.expert_rating_file,
            fibr_dir=args.fibr_dir,
            xgb_model_dir=args.xgb_model_dir,
            output_dir=args.output_dir,
            fig_dir=args.fig_dir,
            random_state=args.random_state,
            image_type=args.image_type,
        )

        compute_irr_with_xgb_rater(
            expert_qc_dir=args.raw_expert_dir,
            fibr_deriv_dir=args.output_dir,
            fig_dir=args.fig_dir,
        )

        plot_xgb_scatter(
            expert_rating_file=args.expert_rating_file,
            output_dir=args.output_dir,
            fibr_dir=args.fibr_dir,
            fig_dir=args.fig_dir,
        )
