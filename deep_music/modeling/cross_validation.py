from sklearn.model_selection import KFold


def cross_validation(df, genres, k=5, seed=77):
    splits = [{'train': list(), 'test': list()}] * 5
    for genre in genres:
        kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

        to_split_df = df[df['track_genre_top'] == genre]
        i = 0
        for train_indx, test_indx in kfold.split(to_split_df):
            train_indx = to_split_df.iloc[train_indx].index
            test_indx = to_split_df.iloc[test_indx].index
            splits[i]['train'].extend(train_indx)
            splits[i]['test'].extend(test_indx)
            i += 1
    return splits
