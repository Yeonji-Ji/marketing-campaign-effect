import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# x_model = fit_xlearner(X, T, y)
# cate_predictions = predict_cate(x_model, X)


df_lift = pd.DataFrame({
    'cate': cate_predictions,
    'treatment': df[treatment],
    'outcome': df[outcome]
})

df_lift = df_lift.sort_values(by='cate', ascending=False).reset_index(drop=True)

df_lift['decile'] = pd.qcut(df_lift.index, q=10, labels=False, duplicates='drop')

grouped = df_lift.groupby('decile')
lift_df = pd.DataFrame()
lift_df['n_treatment'] = grouped['treatment'].sum()
lift_df['n_control'] = grouped['treatment'].count() - grouped['treatment'].sum()
lift_df['outcome_treatment'] = grouped.apply(lambda df: df.loc[df['treatment'] == 1, 'outcome'].sum())
lift_df['outcome_control'] = grouped.apply(lambda df: df.loc[df['treatment'] == 0, 'outcome'].sum())

lift_df['mean_outcome_treatment'] = lift_df['outcome_treatment'] / lift_df['n_treatment'].replace(0, 1)
lift_df['mean_outcome_control'] = lift_df['outcome_control'] / lift_df['n_control'].replace(0, 1)

lift_df['uplift'] = lift_df['mean_outcome_treatment'] - lift_df['mean_outcome_control']

lift_df['total_uplift_effect'] = lift_df['uplift'] * lift_df['n_treatment']

lift_df['cumulative_n_treatment'] = lift_df['n_treatment'].cumsum()
lift_df['cumulative_uplift_effect'] = lift_df['total_uplift_effect'].cumsum()
lift_df['cumulative_mean_uplift'] = lift_df['cumulative_uplift_effect'] / lift_df['cumulative_n_treatment']


# Plot 1
sns.set_style('whitegrid')

plt.figure(figsize=(8, 5))

sns.histplot(cate_predictions, kde=True, bins=50, color='skyblue',)

plt.axvline(x=0, color='red', linestyle='--', linewidth=2)

plt.title('Distribution of Predicted CATE Scores', fontsize=16)
plt.xlabel('Predicted Uplift (CATE Score)')
plt.ylabel('Number of Customers')
y_axis_max = plt.ylim()[1]
plt.text(10, y_axis_max * 0.9, 'Persuadables →', ha='left', color='green', fontsize=12)
plt.text(-10, y_axis_max * 0.9, '← Sleeping Dogs', ha='right', color='red', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_PATH + "/predicted_CATE_distribution.png", dpi=200, bbox_inches='tight')
plt.show()


# Plot 2
plt.figure(figsize=(8, 5))
x_values = np.arange(1, 11)
y_values = lift_df['cumulative_mean_uplift']
plt.plot(x_values, y_values, marker='o', mfc='blue',
         label='My Uplift Model', linewidth=2.5, color='skyblue')

for x, y in zip(x_values, y_values):
    plt.text(x, y + 15, f'{int(y)}', ha='center', fontsize=12, color='black')


overall_uplift = (df_lift[df_lift['treatment'] == 1]['outcome'].mean() - 
                  df_lift[df_lift['treatment'] == 0]['outcome'].mean())
plt.plot(np.arange(1, 11), [overall_uplift] * 10, linestyle='--', color='red', label='Random Targeting')

plt.ylim([360, 790])
plt.title('Cumulative CATE by Ranked Deciles', fontsize=16)
plt.xlabel('Population Targeted (%) (Sorted by Descending CATE Score)', fontsize=14)
plt.ylabel('Cumulative Average Uplift', fontsize=14)
plt.xticks(np.arange(1, 11), [f'{i*10}%' for i in range(1, 11)])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_PATH + "/Uplift_Curve.png", dpi=200, bbox_inches='tight')
plt.show()