#!/usr/bin/env python
import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

SEED = 42

EXCLUDE_TRAIN = ("QueryId", "Label")
EXCLUDE_TEST = ("ObjectId", "QueryId")

NORM_BASES = (
    "bm25f",
    "bm25_title",
    "bm25_body",
    "bm25_url",
    "bm25_stem_title",
    "bm25_stem_body",
    "coverage_title",
    "coverage_body",
    "coverage_url",
    "idf_coverage_title",
    "idf_coverage_body",
    "lm_title",
    "lm_body",
    "tf_sum_title",
    "tf_sum_body",
    "tf_sum_url",
)

LOG_BASES = (
    "bm25f",
    "bm25_title",
    "bm25_body",
    "bm25_url",
    "bm25_stem_title",
    "bm25_stem_body",
    "tf_sum_title",
    "tf_sum_body",
    "tf_sum_url",
    "lm_title",
    "lm_body",
    "domain_docs",
    "domain_train_pairs",
    "title_len",
    "body_len",
    "url_len",
    "url_tokens",
    "url_depth",
    "coverage_title",
    "coverage_body",
    "coverage_url",
    "idf_coverage_title",
    "idf_coverage_body",
    "idf_coverage_url",
    "body_min_window",
    "body_first_pos",
)

PARAMS = dict(
    loss_function="YetiRank",
    iterations=600,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    thread_count=8,
    verbose=False,
    allow_writing_files=False,
)


def add_query_features(df):
    grouped = df.groupby("QueryId", sort=False)
    new_cols = {}

    for base in NORM_BASES:
        if base not in df.columns:
            continue
        mean = grouped[base].transform("mean")
        std = grouped[base].transform("std").replace(0.0, np.nan)
        new_cols[f"{base}__z"] = ((df[base] - mean) / std).fillna(0.0).astype(np.float32)

    for base in LOG_BASES:
        if base not in df.columns:
            continue
        log_values = np.log1p(df[base].clip(lower=0))
        log_grouped = log_values.groupby(df["QueryId"], sort=False)
        mean = log_grouped.transform("mean")
        std = log_grouped.transform("std").replace(0.0, np.nan)
        new_cols[f"{base}__logz"] = (
            (log_values - mean) / std
        ).fillna(0.0).astype(np.float32)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def feature_names(df, exclude):
    return [column for column in df.columns if column not in exclude]


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    data = add_query_features(data)
    features = feature_names(data, EXCLUDE_TRAIN)

    pool = Pool(
        data[features],
        label=data["Label"].astype(int),
        group_id=data["QueryId"],
    )

    model = CatBoostRanker(**PARAMS)
    model.fit(pool)
    model.save_model(model_file)
    print(f"trained on {len(data)} rows, {data['QueryId'].nunique()} queries, "
          f"{len(features)} features -> {model_file}")


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    original = test.copy()
    test = add_query_features(test)
    features = feature_names(test, EXCLUDE_TEST)

    model = CatBoostRanker()
    model.load_model(model_file)

    scores = model.predict(test[features])
    submission = pd.DataFrame({"ObjectId": original["ObjectId"], "Score": scores})
    submission.to_csv(submission_file, index=False)
    print(f"wrote {len(submission)} rows -> {submission_file}")


def main():
    parser = argparse.ArgumentParser(description="Learning to rank homework solution")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--model_file", required=True, help="model file")
    parser.add_argument("--submission_file", help="output submission csv")
    parser.add_argument("data_dir", help="competition data directory")
    args = parser.parse_args()

    start = timer()
    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error("--submission_file is required without --train")
        predict(args.data_dir, args.model_file, args.submission_file)
    print(f"finished, elapsed = {timer() - start:.3f}")


if __name__ == "__main__":
    main()
