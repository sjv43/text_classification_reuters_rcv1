import numpy as np
import pandas as pd

def getReutersData(validation=True, testOnly=False, testSize=10000):
    from sklearn.datasets import fetch_rcv1
    rcv1 = fetch_rcv1()
    
    train_id_filename = 'C:\\Users\\SJV (Work)\\Desktop\\Work\\Reuters\\rcv1_data\\rcv1-1m.id'
    train_id = np.loadtxt(train_id_filename, dtype=np.int64)
    train_index = [np.where(rcv1.sample_id == x)[0][0] for x in train_id]
    X_train = rcv1.data[train_index].toarray()

    train_label_filename = 'C:\\Users\\SJV (Work)\\Desktop\\Work\\Reuters\\rcv1_data\\rcv1-1m.lvl2'
    train_label = np.loadtxt(train_label_filename, dtype='str')
    y_train = np.array([np.where(rcv1.target_names == x)[0][0] for x in train_label])
    
    minFrequency = 2
    freq_cnt = pd.Series(y_train).value_counts(normalize=False)
    unique_train = freq_cnt[freq_cnt >= minFrequency].index
    selected_idx_train = [y_train[i] in unique_train for i in range(len(y_train))]
    X_train = X_train[selected_idx_train]
    y_train = y_train[selected_idx_train]

    test_id_filename = 'C:\\Users\\SJV (Work)\\Desktop\\Work\\Reuters\\rcv1_data\\rcv1-test.id'
    test_id = np.loadtxt(test_id_filename, dtype=np.int64)
    testSize = min(testSize, len(test_id))
    random_test_idx = np.random.choice(range(len(test_id)), size=testSize, replace=False)
    test_index = [np.where(rcv1.sample_id == x)[0][0] for x in test_id[random_test_idx]]
    #test_index = [np.where(rcv1.sample_id == x)[0][0] for x in test_id]
    X_test = rcv1.data[test_index].toarray()

    test_label_filename = 'C:\\Users\\SJV (Work)\\Desktop\\Work\\Reuters\\rcv1_data\\rcv1-test.lvl2'
    test_label = np.loadtxt(test_label_filename, dtype='str')
    y_test = np.array([np.where(rcv1.target_names == x)[0][0] for x in test_label[random_test_idx]])
    #y_test = np.array([np.where(rcv1.target_names == x)[0][0] for x in test_label])
    
    selected_idx = [y_test[i] in unique_train for i in range(len(y_test))]
    X_test = X_test[selected_idx]
    y_test = y_test[selected_idx]
    
    y_train = np.asarray([np.where(unique_train == y)[0][0] for y in y_train])
    y_test = np.asarray([np.where(unique_train == y)[0][0] for y in y_test])
    
    if testOnly == True:
        return X_test, y_test
    elif validation == True:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=2000, random_state=43)
        for train_index, valid_index in sss.split(X_train, y_train):
            X_train, X_valid = X_train[train_index], X_train[valid_index]
            y_train, y_valid = y_train[train_index], y_train[valid_index]
        return X_train, X_test, X_valid, y_train, y_test, y_valid
    else:
        return X_train, X_test, y_train, y_test

X_train, X_test, X_valid, y_train, y_test, y_valid = getReutersData(testSize=2500)