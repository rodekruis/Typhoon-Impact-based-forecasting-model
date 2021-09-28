def splitting_train_test(df):

    # To save the train and test sets
    df_train_list = []
    df_test_list = []

    # List of years that are to be used as a test set
    years = [2016, 2017, 2018, 2019, 2020]

    for year in years:

        df_train_list.append(df[df["year"] < year])
        df_test_list.append(df[df["year"] == year])

    return df_train_list, df_test_list
