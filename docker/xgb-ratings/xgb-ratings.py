#!/opt/conda/bin/python

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import pandas as pd
import re
import s3fs
import shap

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve, roc_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from skopt import BayesSearchCV
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from xgboost.core import XGBoostError


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
            "animated" if not s.endswith("_5") else "static" for s in df_valid_votes["sample"]
        ]
        
        if image_type in ["animated", "static"]:
            df_valid_votes = df_valid_votes[df_valid_votes["image_type"] == image_type]
        elif image_type == "both":
            df_valid_votes["user"] = df_valid_votes["user"] + "_" + df_valid_votes["image_type"]
        else:
            raise ValueError(f"image_type must be 'animated', 'static', 'both', or None. Got {image_type} instead.")
        
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
    # Save these qc labels in a participants.tsv file
    fs = s3fs.S3FileSystem(anon=True)
    dwiqc_s3_uri = (
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/dwiqc.json"
    )
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

    tsv_filename = "participants"
    if image_type is not None:
        tsv_filename += "_" + image_type

    df_participants.to_csv(
        op.join(output_dir, f"{tsv_filename}.tsv"), sep="\t", na_rep="n/a"
    )


def xgb_qc(expert_rating_file, fibr_dir, xgb_model_dir, output_dir, random_state=42, image_type=None):
    # Create the xgb model directory if it doesn't already exist
    os.makedirs(op.abspath(xgb_model_dir), exist_ok=True)

    X, y, df_qc, fibr_votes = _get_Xy(
        expert_rating_file=expert_rating_file, fibr_dir=fibr_dir, image_type=image_type
    )

    Xqc = pd.DataFrame(index=y.index).merge(
        df_qc,
        how="left",
        left_index=True,
        right_index=True,
    ).to_numpy()

    Xfibr = pd.DataFrame(index=y.index).merge(
        fibr_votes,
        how="left",
        left_index=True,
        right_index=True
    ).to_numpy()

    X_columns = X.columns
    y = y.to_numpy()
    X = X.to_numpy()

    print(X.shape)

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
    shap_values = []
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (train, test) in enumerate(tqdm(cv.split(X, y), total=6)):
        try:
            xgb = XGBClassifier()
            xgb.load_model(
                op.join(
                    xgb_model_dir, f"{model_name}_seed-{random_state}_cv-{i}.json"
                )
            )
            checkpointed[i] = True
            models[i] = xgb
            explainer = shap.Explainer(xgb, feature_names=X_columns)
            shap_values.append(explainer(X[test]))
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
                op.join(
                    xgb_model_dir, f"{model_name}_seed-{random_state}_cv-{i}.json"
                )
            )
            checkpointed[i] = False
            explainer = shap.Explainer(
                models[i].best_estimator_, feature_names=X_columns
            )
            shap_values.append(explainer(X[test]))

        try:
            xgb = XGBClassifier()
            xgb.load_model(
                op.join(
                    xgb_model_dir, f"{qc_model_name}_seed-{random_state}_cv-{i}.json"
                )
            )
            qc_models[i] = xgb
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
        ["Fibr + dwiqc", "dwiqc only", "Fibr only"]
    ):
        mean_tpr = np.mean(_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(_aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            label=r"%s (AUC = %0.2f $\pm$ %0.2f)" % (label, mean_auc, std_auc),
            lw=2,
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
            label=r"$\pm$ 1 std. dev." if label=="Fibr + dwiqc" else None,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color=colors[3], label="Chance", alpha=0.8)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Mean receiver operating characteristic (ROC)", fontsize=14)

    plot_title = "xgb_roc_curve"
    shap_csv_title = "xgb_shap_values"
    qc_csv_title = "qc_ratings"
    if image_type is not None:
        plot_title += "_" + image_type
        shap_csv_title += "_" + image_type
        qc_csv_title += "_" + image_type

    fig.savefig(op.join(output_dir, f"{plot_title}.pdf"), bbox_inches="tight")

    absmean_shap = pd.Series(
        index=X_columns,
        data=np.absolute(np.concatenate([sh.values for sh in shap_values])).mean(
            axis=0
        ),
        name="mean-absolute-shap-value",
    )
    absmean_shap.index.rename("feature", inplace=True)
    absmean_shap.sort_values(ascending=False, inplace=True)
    absmean_shap.to_csv(op.join(output_dir, f"{shap_csv_title}.csv"))

    # Create a voting classifier from each CV's XGBoost classifier.
    # Weight each classifier by its out-of-sample ROC AUC
    estimators = {
        f"cv{i}": models[i] if checkpointed[i] else models[i].best_estimator_
        for i in range(6)
    }

    weights = aucs
    voter = VotingClassifier(estimators=estimators, weights=weights, voting="soft")

    # We don't want to refit so manually set the fitted estimators
    voter.estimators_ = list(estimators.values())
    voter.le_ = LabelEncoder().fit(y)
    voter.classes_ = voter.le_.classes_

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
    qc_ratings.to_csv(op.join(output_dir, f"{qc_csv_title}.csv"))

    _save_participants_tsv(qc_ratings, output_dir, image_type=image_type)


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
        "output_dir",
        help="Output directory.",
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
            "statis image and animated image scores are averaged together. Use "
            "'static' or 'animated' to limit fibr ratings to only static or "
            "animated image types respectively. Use 'both' to compute separate "
            "mean ratings for each image type."
        ),
    )

    args = parser.parse_args()

    xgb_qc(
        expert_rating_file=args.expert_rating_file,
        fibr_dir=args.fibr_dir,
        xgb_model_dir=args.xgb_model_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        image_type=args.image_type,
    )
