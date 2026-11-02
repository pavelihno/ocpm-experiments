import pandas as pd

from utils.constants import EXECUTION_ID, OCEL_TIMESTAMP, REMAINING_TIME


def create_unbiased_ocel_split(
    dataset,
    time_col=OCEL_TIMESTAMP,
    case_col=EXECUTION_ID,
    test_share=0.2,
    drop_longest_percentile=0.95,
):
    """
    Implements strict temporal splitting and debiasing for predictive process monitoring.

    This function ensures that training cases complete before test cases start,
    preventing data leakage. It also debiases the dataset's end and the test set's
    beginning by accounting for case duration distributions.

    Reference:
    @article{https://doi.org/10.13140/rg.2.2.16036.19848,
      doi = {10.13140/RG.2.2.16036.19848},
      url = {https://www.researchgate.net/doi/10.13140/RG.2.2.16036.19848},
      author = {Weytjens, Hans and De Weerdt, Jochen},
      title = {Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring},
      year = {2021}
    }
    """
    df = dataset.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Calculate target: Remaining Time per event within an execution
    case_ends = df.groupby(case_col)[time_col].transform('max')
    df[REMAINING_TIME] = (case_ends - df[time_col]).dt.total_seconds()

    # Calculate execution durations and boundaries
    agg_df = df.groupby(case_col)[time_col].agg(['min', 'max'])
    agg_df['duration'] = (agg_df['max'] - agg_df['min']).dt.total_seconds()

    # Remove longest cases to prevent dataset collapse during end-debiasing
    max_duration_threshold = agg_df['duration'].quantile(drop_longest_percentile)
    valid_cases = agg_df[agg_df['duration'] <= max_duration_threshold].index
    df = df[df[case_col].isin(valid_cases)]
    agg_df = agg_df.loc[valid_cases]

    # Determine strict temporal separation point
    starts_sorted = agg_df['min'].sort_values()
    split_idx = int(len(starts_sorted) * (1 - test_share))
    separation_time = starts_sorted.iloc[split_idx]

    # Construct Training Set (Strict Splitting)
    train_cases = agg_df[agg_df['max'] < separation_time].index
    df_train = df[df[case_col].isin(train_cases)].copy()

    # Construct and Debias Test Set
    test_cases_all = agg_df[agg_df['max'] >= separation_time].index
    df_test = df[df[case_col].isin(test_cases_all)].copy()

    # Debias test set start: Drop events that occurred before the separation time
    df_test = df_test[df_test[time_col] >= separation_time].copy()

    # Drop events in the final time window equal to the max case duration
    latest_timestamp = df[time_col].max()
    debias_cutoff = latest_timestamp - pd.Timedelta(seconds=max_duration_threshold)
    df_test = df_test[df_test[time_col] <= debias_cutoff].copy()

    return df_train, df_test
