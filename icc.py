import os
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.stats import chi2
from scipy.stats import rankdata


# ==========================================================
# 1) File settings
# ==========================================================

file_path = r"weights_by_experts.xlsx"
sheet_name = "Sheet2"

output_file = r"icc_results.xlsx"


# ==========================================================
# 2) Read Excel file
# ==========================================================
# header=None is essential because the first row is also data.
# This prevents losing the first relation.
# ==========================================================

raw_df = pd.read_excel(
    file_path,
    sheet_name=sheet_name,
    header=None
)

print("\n--- Raw Excel data preview ---")
print(raw_df.head())


# ==========================================================
# 3) Detect whether the first column contains relation labels
# ==========================================================

# Try to convert the first column to numeric values
first_column_numeric = pd.to_numeric(
    raw_df.iloc[:, 0],
    errors="coerce"
)

# If the first column is mostly non-numeric, it is assumed
# to contain relation labels such as R1, R2, ..., R34.
non_numeric_first_column = first_column_numeric.isna().sum()
total_rows = len(raw_df)

has_relation_label_column = (
    non_numeric_first_column > total_rows / 2
)

if has_relation_label_column:
    relation_labels = raw_df.iloc[:, 0].astype(str).str.strip()
    rating_df = raw_df.iloc[:, 1:].copy()
else:
    relation_labels = pd.Series(
        [f"R{i+1}" for i in range(len(raw_df))]
    )
    rating_df = raw_df.copy()


# ==========================================================
# 4) Convert ratings to numeric values
# ==========================================================

data = rating_df.apply(pd.to_numeric, errors="coerce")

# Remove completely empty rows and columns
non_empty_rows = ~data.isna().all(axis=1)
non_empty_columns = ~data.isna().all(axis=0)

data = data.loc[non_empty_rows, non_empty_columns].copy()
relation_labels = relation_labels.loc[data.index].reset_index(drop=True)
data = data.reset_index(drop=True)


# Generate dynamic expert names
data.columns = [
    f"Expert_{i+1}" for i in range(data.shape[1])
]

data.index = relation_labels


print("\n--- Data dimensions ---")
print(f"Number of relations (rows): {data.shape[0]}")
print(f"Number of experts (columns): {data.shape[1]}")


# ==========================================================
# 5) Data-quality checks
# ==========================================================

missing_values = int(data.isna().sum().sum())

if missing_values > 0:
    print(f"\nWarning: {missing_values} missing value(s) found.")

    missing_rows = data.index[data.isna().any(axis=1)].tolist()

    print("Relations containing missing values:")
    print(missing_rows)

    print(
        "\nRows with missing values will be excluded "
        "from ICC and Kendall's W calculations."
    )

    complete_data = data.dropna(axis=0).copy()

else:
    complete_data = data.copy()


# Check the expected normalized scale: -1 to 1
out_of_range_mask = (
    (complete_data < -1) | (complete_data > 1)
)

out_of_range_values = int(out_of_range_mask.sum().sum())

if out_of_range_values > 0:
    print(
        "\nWARNING: Some values are outside the expected "
        "0–1 scale."
    )

    print(
        "Please verify whether the ratings were entered "
        "using the correct normalized scale."
    )

else:
    print("\nScale check passed: all values are within -1 – 1.")


if complete_data.shape[0] < 2:
    raise ValueError(
        "At least two complete relations are required."
    )

if complete_data.shape[1] < 2:
    raise ValueError(
        "At least two experts are required."
    )


print("\n--- Complete data used for reliability analysis ---")
print(f"Complete relations: {complete_data.shape[0]}")
print(f"Experts: {complete_data.shape[1]}")


# ==========================================================
# 6) Convert data to long format for Pingouin
# ==========================================================

long_data = (
    complete_data
    .reset_index(names="Relation")
    .melt(
        id_vars="Relation",
        var_name="Expert",
        value_name="Score"
    )
)

print("\n--- Long-format preview ---")
print(long_data.head())


# ==========================================================
# 7) Calculate ICC
# ==========================================================

icc_table = pg.intraclass_corr(
    data=long_data,
    targets="Relation",
    raters="Expert",
    ratings="Score"
)

icc_table = icc_table.copy()

print("\n--- Full ICC output ---")
print(icc_table.to_string(index=False))


# Pingouin versions may use CI95 or CI95%
if "CI95" in icc_table.columns:
    ci_column = "CI95"
elif "CI95%" in icc_table.columns:
    ci_column = "CI95%"
else:
    ci_column = None
    print("\nWarning: Confidence interval column was not found.")


# Normalize ICC type names
icc_table.loc[:, "Type_normalized"] = (
    icc_table["Type"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# In Pingouin, ICC(A,k) corresponds to ICC(2,k):
# two-way random effects, absolute agreement, average measures.
result_rows = icc_table[
    icc_table["Type_normalized"] == "ICC(A,K)"
].copy()


if result_rows.empty:
    raise ValueError(
        "\nICC(A,k), corresponding to ICC(2,k), "
        "was not found in Pingouin output."
    )


result = result_rows.iloc[0]


# ==========================================================
# 8) Display ICC(2,k)
# ==========================================================

print("\n" + "=" * 65)
print(
    "ICC(2,k): Two-way random effects, "
    "absolute agreement, average measures"
)
print("=" * 65)

print(f"ICC(2,k): {result['ICC']:.4f}")

if ci_column is not None:
    print(f"95% Confidence Interval: {result[ci_column]}")

print(f"F-value: {result['F']:.4f}")
print(
    f"Degrees of freedom: "
    f"df1 = {result['df1']}, df2 = {result['df2']}"
)
print(f"p-value: {result['pval']:.6e}")


# ==========================================================
# 9) Calculate Kendall's W
# ==========================================================

def kendalls_w(dataframe):
    """
    Calculate Kendall's coefficient of concordance W.

    Rows    = targets/relations
    Columns = raters/experts
    """

    matrix = dataframe.to_numpy(dtype=float)

    n_targets, n_raters = matrix.shape

    if n_targets < 2 or n_raters < 2:
        raise ValueError(
            "Kendall's W requires at least two targets "
            "and two raters."
        )

    # Rank each expert's scores across all relations.
    # Average ranks are used for ties.
    ranked_matrix = np.apply_along_axis(
        rankdata,
        axis=0,
        arr=matrix,
        method="average"
    )

    rank_sums = ranked_matrix.sum(axis=1)
    mean_rank_sum = rank_sums.mean()

    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    # Tie correction for each expert
    tie_correction = 0

    for expert_index in range(n_raters):
        values, counts = np.unique(
            matrix[:, expert_index],
            return_counts=True
        )

        tied_groups = counts[counts > 1]

        tie_correction += np.sum(
            tied_groups**3 - tied_groups
        )

    denominator = (
        n_raters**2 * (n_targets**3 - n_targets)
        - n_raters * tie_correction
    )

    if denominator == 0:
        return np.nan, np.nan, np.nan, np.nan

    W = (12 * S) / denominator

    # Test statistic for Kendall's W
    chi_square = n_raters * (n_targets - 1) * W
    degrees_of_freedom = n_targets - 1
    p_value = chi2.sf(
        chi_square,
        degrees_of_freedom
    )

    return (
        W,
        chi_square,
        degrees_of_freedom,
        p_value
    )


kendall_w, kendall_chi2, kendall_df, kendall_p = (
    kendalls_w(complete_data)
)


print("\n" + "=" * 65)
print("Kendall's Coefficient of Concordance")
print("=" * 65)

print(f"Kendall's W: {kendall_w:.4f}")
print(f"Chi-square: {kendall_chi2:.4f}")
print(f"Degrees of freedom: {kendall_df}")
print(f"p-value: {kendall_p:.6e}")


# ==========================================================
# 10) Descriptive statistics for each relation
# ==========================================================

descriptive_stats = pd.DataFrame({
    "Relation": complete_data.index,
    "N_Experts": complete_data.notna().sum(axis=1),
    "Mean": complete_data.mean(axis=1),
    "Median": complete_data.median(axis=1),
    "Std_Dev": complete_data.std(axis=1, ddof=1),
    "IQR": (
        complete_data.quantile(0.75, axis=1)
        - complete_data.quantile(0.25, axis=1)
    ),
    "Minimum": complete_data.min(axis=1),
    "Maximum": complete_data.max(axis=1),
    "Range": (
        complete_data.max(axis=1)
        - complete_data.min(axis=1)
    )
})

descriptive_stats = descriptive_stats.reset_index(drop=True)

print("\n--- Descriptive statistics preview ---")
print(descriptive_stats.head())


# ==========================================================
# 11) Summary table
# ==========================================================

icc_ci_value = (
    result[ci_column]
    if ci_column is not None
    else "Not available"
)

summary = pd.DataFrame({
    "Statistic": [
        "Model",
        "Number of relations in original data",
        "Number of complete relations used",
        "Number of experts",
        "Missing values",
        "Scale",
        "ICC(2,k)",
        "95% CI",
        "F-value",
        "ICC df1",
        "ICC df2",
        "ICC p-value",
        "Kendall's W",
        "Kendall's Chi-square",
        "Kendall's df",
        "Kendall's p-value"
    ],
    "Value": [
        (
            "Two-way random effects; absolute agreement; "
            "average measures"
        ),
        data.shape[0],
        complete_data.shape[0],
        complete_data.shape[1],
        missing_values,
        "-1 – 1 normalized scale",
        result["ICC"],
        str(icc_ci_value),
        result["F"],
        result["df1"],
        result["df2"],
        result["pval"],
        kendall_w,
        kendall_chi2,
        kendall_df,
        kendall_p
    ]
})


# ==========================================================
# 12) Save all results in one Excel workbook
# ==========================================================

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    icc_table.to_excel(
        writer,
        sheet_name="All_ICC_Models",
        index=False
    )

    descriptive_stats.to_excel(
        writer,
        sheet_name="Descriptive_Stats",
        index=False
    )

    data.reset_index(names="Relation").to_excel(
        writer,
        sheet_name="Cleaned_All_Data",
        index=False
    )

    complete_data.reset_index(names="Relation").to_excel(
        writer,
        sheet_name="Complete_Data_Used",
        index=False
    )

print("\n" + "=" * 65)
print("All analyses completed successfully.")
print(f"Results saved to:\n{output_file}")
print("=" * 65)
