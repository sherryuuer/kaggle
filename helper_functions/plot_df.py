import matplotlib.pyplot as plt
import seaborn as sns


def plot_numerics(df):
    numeric_columns = df.columns
    # Plotting histograms and box plots for each numeric column
    for column in numeric_columns:
        _, ax = plt.subplots(1, 2, figsize=(16, 5))
        ax = ax.flatten()

        # Histogram
        sns.histplot(df[column], bins=50, kde=True,
                     color='skyblue', ax=ax[0])
        ax[0].set_title(f'Histogram of {column}',
                        fontsize=15, fontweight='bold')
        ax[0].set_xlabel(column, fontsize=12)
        ax[0].set_ylabel('Frequency', fontsize=12)

        # Box plot
        sns.boxplot(x=df[column], color='lightgreen', ax=ax[1])
        ax[1].set_title(f'Box plot of {column}',
                        fontsize=15, fontweight='bold')
        ax[1].set_xlabel(column, fontsize=12)

        plt.tight_layout()
        plt.show()

# example
# df_train["Whole weight Ratio"]=df_train["Whole weight.1"]/df_train["Whole weight"]
# cat_cols=['Sex']
# num_cols=['Length', 'Diameter', 'Height',
#              'Whole weight','Whole weight.1',
#              'Whole weight.2', 'Shell weight', "Whole weight Ratio"]
# plot_numerics(df_train[num_cols])


def plot_cat(df_train, cat, figsize=(25, 12)):
    plt.figure(figsize=figsize)
    # cat means categorical data
    ax = sns.countplot(x=df_train[cat],
                       order=df_train[cat].value_counts(ascending=False).index)

    abs_values = df_train[cat].value_counts(ascending=False)  # 绝对计数
    rel_values = df_train[cat].value_counts(
        ascending=False, normalize=True).values * 100  # 相对频率
    lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(
        abs_values, rel_values)]  # 类别标签列表

    ax.bar_label(container=ax.containers[0], labels=lbls)
    ax.set_title("Distribution of "+cat+" Values", fontsize=16)

# example
# get_count_plot('Rings')


def plot_corr_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, vmax=1, vmin=0.5, center=0.75, annot=True, fmt=".2f", square=True,
                linewidths=.5, cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features', fontsize=15)
    plt.xticks(fontsize=8, fontweight='bold')
    plt.yticks(fontsize=8, fontweight='bold')
    plt.show()

# example
# plot_corr_matrix(df_train[num_cols+['Rings']])
