# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from  itertools import product
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import datetime

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

N_train = 2308
N_val = 524
N_test = 20
TIME_FRAME = 11
PREDICTION_TIMESTEPS = 30
X_columns =40*TIME_FRAME

UNNEEDED_COLUMNS = ['time step',
                    ' id0', ' id1', ' id2', ' id3', ' id4', ' id5', ' id6', ' id7', ' id8', ' id9',
                    ' type0', ' type1', ' type2', ' type3', ' type4', ' type5', ' type6', ' type7', ' type8', ' type9']

def load_data():
    df_X_train, df_y_x_train, df_y_y_train = load_train_data()
    df_X_val, df_y_x_val, df_y_y_val = load_validation_data()
    df_X_test = load_test_data()

    return df_X_train, df_y_x_train, df_y_y_train, df_X_val, df_y_x_val, df_y_y_val, df_X_test

def load_test_data():
    # TEST - X
    df_list = []
    path = '/kaggle/input/cpsc340w20finalpart2/test/X'
    for file in os.listdir(path):
        if file.endswith('.csv'):
            with open(os.path.join(path,file), 'rb') as f:
                df = pd.read_csv(f).drop(UNNEEDED_COLUMNS, axis=1)

                cols = ['{}_{}'.format(a,b) for a, b in product(df.index, df.columns)]
                df = pd.DataFrame([df.values.ravel()], columns=cols)
                df_list.append(df.fillna(0))

    df_X_test = pd.concat(df_list)

    assert df_X_test.shape[0] == N_test
    assert df_X_test.shape[1] == X_columns

    print("[Script] Test dataset loaded.")

    return df_X_test

def load_validation_data():
    # VAL - X
    df_list = []
    path = '/kaggle/input/cpsc340w20finalpart2/val/X'
    for file in os.listdir(path):
        if file.endswith('.csv'):
            with open(os.path.join(path,file), 'rb') as f:
                df = pd.read_csv(f).drop(UNNEEDED_COLUMNS, axis=1)

                cols = ['{}_{}'.format(a,b) for a, b in product(df.index, df.columns)]
                df = pd.DataFrame([df.values.ravel()], columns=cols)
                df_list.append(df.fillna(0))

    df_X_val = pd.concat(df_list)

    # VAL - Y
    df_list_y_x = []
    df_list_y_y = []
    path = '/kaggle/input/cpsc340w20finalpart2/val/y'
    for file in os.listdir(path):
        if file.endswith('.csv'):
            with open(os.path.join(path,file), 'rb') as f:
                # For x coordinate
                df = pd.read_csv(f).drop(['time step', ' y'], axis=1)

                cols = ['{}_{}'.format(a,b) for a, b in product(df.index, df.columns)]
                df = pd.DataFrame([df.values.ravel()], columns=cols)
                df_list_y_x.append(df.fillna(0))

            with open(os.path.join(path,file), 'rb') as f:
                # For y coordinate
                df = pd.read_csv(f).drop(['time step', ' x'], axis=1)

                cols = ['{}_{}'.format(a,b) for a, b in product(df.index, df.columns)]
                df = pd.DataFrame([df.values.ravel()], columns=cols)
                df_list_y_y.append(df.fillna(0))

    df_y_x_val = pd.concat(df_list_y_x)
    df_y_y_val = pd.concat(df_list_y_y)

    assert df_X_val.shape[0] == N_val
    assert df_X_val.shape[1] == X_columns

    assert df_y_x_val.shape == df_y_y_val.shape
    assert df_y_x_val.shape[0] == N_val
    assert df_y_x_val.shape[1] == 30

    print("[Script] Validation dataset loaded.")

    return df_X_val, df_y_x_val, df_y_y_val

def load_train_data():
    # TRAIN - X
    df_list = []
    path = '/kaggle/input/cpsc340w20finalpart2/train/X'
    for file in os.listdir(path):
        if file.endswith('.csv'):
            with open(os.path.join(path,file), 'rb') as f:
                df = pd.read_csv(f).drop(UNNEEDED_COLUMNS, axis=1)

                cols = ['{}_{}'.format(a,b) for a, b in product(df.index, df.columns)]
                df = pd.DataFrame([df.values.ravel()], columns=cols)
                df_list.append(df.fillna(0))

    df_X_train = pd.concat(df_list)

    # TRAIN - Y
    df_list_y_x = []
    df_list_y_y = []
    path = '/kaggle/input/cpsc340w20finalpart2/train/y'
    for file in os.listdir(path):
        if file.endswith('.csv'):
            with open(os.path.join(path,file), 'rb') as f:
                # For x coordinate
                df = pd.read_csv(f).drop(['time step', ' y'], axis=1)

                cols = ['{}_{}'.format(a,b) for a, b in product(df.index, df.columns)]
                df = pd.DataFrame([df.values.ravel()], columns=cols)
                df_list_y_x.append(df.fillna(0))

            with open(os.path.join(path,file), 'rb') as f:
                # For y coordinate
                df = pd.read_csv(f).drop(['time step', ' x'], axis=1)

                cols = ['{}_{}'.format(a,b) for a, b in product(df.index, df.columns)]
                df = pd.DataFrame([df.values.ravel()], columns=cols)
                df_list_y_y.append(df.fillna(0))

    df_y_x_train = pd.concat(df_list_y_x)
    df_y_y_train = pd.concat(df_list_y_y)

    # ASSERTIONS
    assert df_X_train.shape[0] == N_train
    assert df_X_train.shape[1] == X_columns

    assert df_y_x_train.shape == df_y_y_train.shape
    assert df_y_x_train.shape[0] == N_train
    assert df_y_x_train.shape[1] == 30

    print("[Script] Train dataset loaded.")

    return df_X_train, df_y_x_train, df_y_y_train

# Only for train
def drop_sparse_intersection(df_X_train, df_y_x_train, df_y_y_train, k):
    present_cols = df_X_train.loc[:, df_X_train.columns.str.startswith('0_ present')].values
    condition = np.count_nonzero(present_cols, axis=1) > k

    df_X_train = df_X_train[condition]
    df_y_x_train = df_y_x_train[condition]
    df_y_y_train = df_y_y_train[condition]

    return df_X_train, df_y_x_train, df_y_y_train

def dist(agent_x, agent_y, x, y):
    # The distance at the start of the time frame
    return np.sqrt((agent_x[:,0][:,None]-x[:,::TIME_FRAME])**2 + (agent_y[:,0][:,None]-y[:,::TIME_FRAME])**2)

def sort_objects(df_X_train):
    # 1. Make a new dataframe, and put the agent's x and y in the first two columns
    # 2. For the rest of the objects, sort in ascending distance

    df_X_train_value = df_X_train.values
    N, D = df_X_train_value.shape

    # Put x & y separately
    agent_x = np.zeros((N,TIME_FRAME))
    agent_y = np.zeros((N,TIME_FRAME))
    x = np.zeros((N,int(D/3-TIME_FRAME))) # now we have three different column types, and need to - TIME_FRAME for agent
    y = np.zeros((N,int(D/3-TIME_FRAME)))

    # Per row has only 11 agent
    for i in range(N):
        temp_agent_x = []
        temp_agent_y = []
        temp_x = []
        temp_y = []

        count = 0
        for j in range(D):
            if df_X_train_value[i][j] == " agent":
                temp_agent_x.append(df_X_train_value[i][j+1])
                temp_agent_y.append(df_X_train_value[i][j+2])

            elif df_X_train_value[i][j] == " others":
                temp_x.append(df_X_train_value[i][j+1])
                temp_y.append(df_X_train_value[i][j+2])
                count += 1

        # There are some 0.0 objects
        gap_num = x.shape[1] - len(temp_x)
        temp_x += [0.0]*gap_num
        temp_y += [0.0]*gap_num

        agent_x[i] = temp_agent_x[:]
        agent_y[i] = temp_agent_y[:]
        x[i] = temp_x[:]
        y[i] = temp_y[:]

    # Calculate the distance, in matrix
    distance = dist(agent_x, agent_y, x, y)

    # Sort the distance each row and get the sorted index
    sort_idx = np.argsort(distance, axis=1)

    sort_idx = sort_idx * 11
    sort_idx_duplicated = np.repeat(sort_idx, TIME_FRAME, axis=1)

    adder = np.arange(TIME_FRAME)
    adder_duplicated = np.tile(adder, int(x.shape[1]/TIME_FRAME))

    idx = sort_idx_duplicated + adder_duplicated[None,:]

    # Insert agent x & y at the start
    # Sort the rest of objects wrt. idx
    sorted_x = np.array(list(map(lambda a, b: b[a], idx, x)))
    sorted_y = np.array(list(map(lambda a, b: b[a], idx, y)))

    agent_sorted_x = np.concatenate((agent_x,sorted_x),axis=1)
    agent_sorted_y = np.concatenate((agent_y,sorted_y),axis=1)

    # Stack xy together: agent_x, agent_y, x0, y0, x1, y1 ...
    stack_xy = np.vstack((agent_sorted_x,agent_sorted_y)).ravel('F').reshape((int(D/3*2),N)).T

    df_X = pd.DataFrame(stack_xy)

    return df_X

def reformatX(df_X):
    df_X_values = df_X.values
    N, D = df_X_values.shape

    ego_car_x = df_X_values[:,::int(D/TIME_FRAME)]
    ego_car_y = df_X_values[:,1::int(D/TIME_FRAME)]

    other = np.delete(df_X_values, np.s_[::int(D/TIME_FRAME)], 1)
    other = np.delete(other, np.s_[::int(D/TIME_FRAME)], 1)

    newdf = np.concatenate((ego_car_x,ego_car_y,other),axis=1)

    # ego_x_t0 ego_x_t1 ... ego_y_t0 ego_y_t1 ... other0_x_t0 other0_y_t0 ...
    df_X = pd.DataFrame(newdf)

    return df_X

def add_ego_car_pos(df_X, df_y_x, df_y_y):
    newdf = pd.DataFrame(np.repeat(df_X.values,PREDICTION_TIMESTEPS,axis=0))
    newdf.columns = df_X.columns
    N, D = newdf.shape

    newdf_yx = pd.DataFrame(np.repeat(df_y_x.values,PREDICTION_TIMESTEPS,axis=0))
    newdf_yx.columns = df_y_x.columns

    newdf_yy = pd.DataFrame(np.repeat(df_y_y.values,PREDICTION_TIMESTEPS,axis=0))
    newdf_yy.columns = df_y_y.columns

    ego_car_x = newdf.values[:,::int(D/TIME_FRAME)]
    ego_car_y = newdf.values[:,1::int(D/TIME_FRAME)]

    other = np.delete(newdf.values, np.s_[::int(D/TIME_FRAME)], 1)
    other = np.delete(other, np.s_[::int(D/TIME_FRAME)], 1)

    all_ego_car_x = np.concatenate((ego_car_x, newdf_yx.values),axis=1)
    all_ego_car_y = np.concatenate((ego_car_y, newdf_yy.values),axis=1)

    all_time = TIME_FRAME + PREDICTION_TIMESTEPS

    time_window_x = np.zeros((N, TIME_FRAME))
    time_window_y = np.zeros((N, TIME_FRAME))

    for i in range(N):
        start = i % PREDICTION_TIMESTEPS
        time_window_x[i] = all_ego_car_x[i][start:start+TIME_FRAME]
        time_window_y[i] = all_ego_car_y[i][start:start+TIME_FRAME]

    # concat
    newdf = np.concatenate((time_window_x,time_window_y,other),axis=1)

    df_X = pd.DataFrame(newdf)

    return df_X

# The y here should have only one column. Select the column at main
def prep_data(df_X_train, df_X_val, df_X_test, df_y_x_train, df_y_y_train):
    df_X_train, df_y_x_train, df_y_y_train = drop_sparse_intersection(df_X_train, df_y_x_train, df_y_y_train, 6)

    df_X_train = df_X_train.loc[:,~df_X_train.columns.str.contains('present')]
    df_X_val = df_X_val.loc[:,~df_X_val.columns.str.contains('present')]
    df_X_test = df_X_test.loc[:,~df_X_test.columns.str.contains('present')]

    df_X_train = sort_objects(df_X_train)
    df_X_val = sort_objects(df_X_val)
    df_X_test = sort_objects(df_X_test)

    df_X_train = add_ego_car_pos(df_X_train, df_y_x_train, df_y_y_train)

    df_X_train = reformatX(df_X_train)
    df_X_val = reformatX(df_X_val)
    df_X_test = reformatX(df_X_test)
#     # Feature selection, wrt to y(x coordiante) and y (y coordinate)
#     num_feature = 100
#     df_X_train_yx = select_feature(df_X_train, df_y_x_train, num_feature)
#     df_X_train_yy = select_feature(df_X_train, df_y_y_train, num_feature)

    return df_X_train, df_X_train, df_X_val, df_X_test, df_y_x_train, df_y_y_train

def score(self, w,X,df_y,criteria):
    if criteria == 'BIC':
        N,D = X.shape
        bias = np.ones((len(X),1))
        Z = np.concatenate((bias, X), axis=1)
        score = 1/2*(np.linalg.norm(np.dot(Z,w)-df_y))**2 + 1/2*np.log(N)*D
    return score

def leastSquares(X,y):
    bias = np.ones((len(X),1))
    Z = np.concatenate((bias, X), axis=1)
    v = solve(Z.T@Z, Z.T@y)
    return v

def forward_selection(df, y, num_feature):
    Count = 0
    df_X = df.values
    df_y = y.values
    N, D = df_X.shape

    # set up an empty set of features
    S = np.array([])
    features = np.arange(D)

    X = np.zeros((N, num_feature))

    while len(S)<num_feature:
        best_score = np.inf
        best_feature = None
        rest_features = np.delete(features,S)

        for f in rest_features:
            X[:,len(S)] = df_X[:,f]
            try:
                w = leastSquares(X[:,:len(S)+2],df_y)
                current_score = score(w,X[:,:len(S)+2],df_y,'BIC')
            except:
                Count = Count +1
                continue
            if(current_score<best_score):
                best_score = current_score
                best_feature = f

        X[:,len(S)] = df_X[:,best_feature]
        S = np.append(S,best_feature)

    return S

def select_feature(df, y, num_feature):
    feature_idx = forward_selection(df, y, num_feature).astype('int')
    column_index_names = df.columns.values[feature_idx]
    df = df[column_index_names]
    return df

class linear_regression():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.w = solve(X.T@X, X.T@y)
#         print(self.w)
    def predict(self, X):
        return X@self.w

if __name__ == "__main__":
    start=datetime.datetime.now()

    # LOAD DATA
    df_X_train, df_y_x_train, df_y_y_train, df_X_val, df_y_x_val, df_y_y_val, df_X_test = load_data()

    # PREP DATA
    df_X_train_yx, df_X_train_yy, df_X_val, df_X_test, df_y_x_train, df_y_y_train = prep_data(df_X_train, df_X_val, df_X_test, df_y_x_train, df_y_y_train)

    # TRAIN
    model_x = linear_regression()
    model_y = linear_regression()

    Xx = df_X_train_yx.values
    Xy = df_X_train_yy.values

    yx = df_y_x_train.values.flatten()
    yy = df_y_y_train.values.flatten()
#     print(Xx.shape)
#     print(Xx[:,22])
#     print(np.linalg.matrix_rank(Xx[:,22]))

    model_x.fit(Xx, yx)
    model_y.fit(Xy, yy)

    # training error:
    yhat_x_tr = model_x.predict(Xx)
    yhat_y_tr = model_y.predict(Xy)
    print(yhat_x_tr)
    trainErrorX = mean_squared_error(yx, yhat_x_tr)
    trainErrorY = mean_squared_error(yy, yhat_y_tr)

    print("Training error for x = ", trainErrorX)
    print("Training error for y = ", trainErrorY)

#     df_X_test = reformatX(df_X_test)

    # In each iteration, (x,y) are predicted separately, and only use 1000ms time frame
    # After prediction, update X & y for testing
    yhat_x_list = []
    yhat_y_list = []
    for t in range(PREDICTION_TIMESTEPS):
        df_X_test_values = df_X_test.values

        yhat_x = model_x.predict(df_X_test_values)
        yhat_y = model_y.predict(df_X_test_values)

        yhat_x_list.append(yhat_x)
        yhat_y_list.append(yhat_y)

        # Update test dataframe
        df_X_test_values[:,1:]
        temp = np.insert(df_X_test_values, TIME_FRAME, yhat_x, 1)
        temp = np.delete(temp, 0, 1)  # delete the first column

        temp = np.insert(df_X_test_values, TIME_FRAME*2, yhat_y, 1)
        temp = np.delete(temp, TIME_FRAME, 1)  # delete the time_frame-th (11th) column
        print(temp.shape)

        df_X_test = pd.DataFrame(temp)


    print("test x: ", yhat_x_list)
    print("test y: ", yhat_y_list)
    print("Elapsed time: ", datetime.datetime.now()-start)
