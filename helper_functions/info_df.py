def info_df(df):
    # get df rows and cols num
    num_rows = df.shape[0]
    num_cols = df.shape[1]

    print("Number of rows:", num_rows)
    print("Number of columns:", num_cols)

    print("\nDataFrame info:")
    print(df.info())

    print("\nDataFrame description:")
    print(df.describe())
