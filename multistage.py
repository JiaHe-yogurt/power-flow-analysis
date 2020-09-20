
def inv_twostage(input_dim,num_bus, num_neural1=40, dropout=0.2):
    inputs = Input(shape=(input_dim,), name='input')

    x = Dense(units=num_neural1, activation='relu', kernel_initializer=initializers.RandomNormal(seed=1),
              bias_initializer=initializers.Zeros())(inputs)

    y_predictions = []
    for bus_idx in range(0, int(num_bus*2)):
        y_prediction = Dense(units=1, activation=None, kernel_initializer=initializers.RandomNormal(seed=1),
                             bias_initializer=initializers.Zeros(), name='y' + str(bus_idx) + '_predictions')(x)
        y_predictions.append(y_prediction)

    concat_pred = Concatenate(1)(y_predictions)

    model = Model(inputs=inputs, outputs=concat_pred)

    return model
def svm_twostage(X_train, X_test, Y_train, Y_test, num_bus, kernel= 'poly', degree=2, epsilon = 0.01):
    y_train, y_test = Y_train[:,:(2*num_bus)], Y_test[:,:(2*num_bus)]
    svr= MultiOutputRegressor(SVR(kernel=kernel,degree=degree, C=1, epsilon=epsilon)).fit(X_train,y_train)
    train_pred = svr.predict(X_train)
    train_rms = np.sqrt(mean_squared_error(y_train, train_pred))
    train_pe_rms = Penalty(Y_train[:, (2 * num_bus):], train_pred, num_bus)
    test_pred = svr.predict(X_test)
    test_rms = np.sqrt(mean_squared_error(y_test, test_pred))
    test_pe_rms = Penalty(Y_test[:, (2 * num_bus):], test_pred, num_bus)
    result = np.array([train_rms,train_pe_rms,test_rms ,test_pe_rms ])
    return result , train_pred , test_pred






# svr
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

num_bus,  Seed, test_size = 39, 1, 0.2


##  multistage, don't normalize X
def inv_prepare_data(num_bus, X_std, Y_std, seed, test_size=0.1):
    train_path = '/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + ' bus'

    V, Theta, P, Q, file_name = inv_load_data(train_path)
    np.random.seed(seed)
    P += np.random.normal(0, np.abs(np.mean(P, axis=0)) * X_std, (P.shape[0], P.shape[1]))
    np.random.seed(seed)
    Q += np.random.normal(0, np.abs(np.mean(Q, axis=0)) * X_std, (Q.shape[0], Q.shape[1]))
    np.random.seed(seed)
    V += np.random.normal(0, np.abs(np.mean(V, axis=0)) * Y_std, (V.shape[0], V.shape[1]))
    np.random.seed(seed)
    Theta += np.random.normal(0, np.abs(np.mean(Theta, axis=0)) * Y_std, (Theta.shape[0], Theta.shape[1]))
    X, Y = np.column_stack([P, Q]), np.column_stack([V, Theta, P, Q])
 #   X = normalize(X, axis=0)  # ,normalize(Q, axis=0)  minmax_scale gives worse reesult
 #   Y = normalize(Y, axis=0)
    train_index, test_index = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=seed)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    Y_bus, ref_bus = construct_Y_bus(num_bus)

    return X_train, X_test, Y_train, Y_test, Y_bus, ref_bus

num_bus=39
Stage=3
X_std = 0.1
Y_std = X_std
kernel, epsilon = (['poly', 'poly', 'poly','poly']), ([0.01,0.01,0.01,0.01])
RR = []  ## columne augment
for Seed in range(3):
          R=[]
          X_train, X_test, Y_train, Y_test, Y_bus, ref_bus = inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size)
          X_train, X_test, Y_train, Y_test = X_train[:100,:], X_test[:20,:], Y_train[:100,:], Y_test[:20,:]
          result, train_pred, test_pred = svm_twostage(X_train, X_test, Y_train, Y_test, num_bus, kernel= kernel[0], degree=2, epsilon = epsilon[0])
          R.append(result)
          for stage in range( Stage):
            X_train_aug, X_test_aug = np.column_stack([X_train, train_pred]), np.column_stack([X_test, test_pred])
            result2, train_pred, test_pred = svm_twostage(X_train_aug, X_test_aug, Y_train, Y_test, num_bus,
                                    kernel=kernel[stage+1], epsilon = epsilon[stage+1] ,degree=2 )
            R.append(result2)
          RR.append(R)
#np.round(np.mean(RR, axis=0),3)

print(np.mean(np.array(RR), axis=0))
#print(np.mean(np.array(RR2), axis=0))
print(np.round(np.mean(np.array(RR), axis=0),3))



## two stage
RR1, RR2 = [], []   ## don't normalize X
for Seed in range(4):
    R1, R2= [], []
    for miss in ([None, 0.1,0.2 ]):
        X_std=0.01
        Y_std =0
        X_train, X_test, Y_train, Y_test, Y_bus, ref_bus = inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size, miss = miss)
        result, train_pred, test_pred = svm_twostage(X_train, X_test, Y_train, Y_test, num_bus, kernel= 'poly', degree=2, epsilon = 0.01)
        R1.append(result)
        X_train_aug, X_test_aug = np.column_stack([X_train, train_pred]), np.column_stack([X_test, test_pred])
        result2, train_pred, test_pred = svm_twostage(X_train_aug, X_test_aug, Y_train, Y_test, num_bus, kernel= 'poly',degree=2, epsilon = 0.01)
        R2.append(result2)
    RR1.append(R1)
    RR2.append(R2)

print(np.mean(np.array(RR1), axis=0))
print(np.mean(np.array(RR2), axis=0))



RR1, RR2 = [], []
for Seed in range(3):
    R1, R2= [], []
    for miss in ([None, 0.01, 0.05,0.1, 0.2 ]):
        X_std=0.01
        Y_std =X_std
        X_train, X_test, Y_train, Y_test, Y_bus, ref_bus= inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size, miss = miss)
        result, train_pred, test_pred = svm_twostage(X_train, X_test, Y_train, Y_test, num_bus, kernel= 'poly', degree=2, epsilon = 0.01)
        R1.append(result)
        X_train_aug, X_test_aug = np.column_stack([X_train, train_pred]), np.column_stack([X_test, test_pred])
        result2, train_pred, test_pred = svm_twostage(X_train_aug, X_test_aug, Y_train, Y_test, num_bus, kernel= 'poly',degree=2, epsilon = 0.01)
        R2.append(result2)
    RR1.append(R1)
    RR2.append(R2)

print(np.mean(np.array(RR1), axis=0))
print(np.mean(np.array(RR2), axis=0))




RR1, RR2 = [], []   ## change train size
for Seed in range(4):
    R1, R2= [], []
    for train_size in ([ 0.8,0.5]):
        X_std=0.005
        Y_std = X_std
        X_train, X_test, Y_train, Y_test, Y_bus, ref_bus = inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size)
        X_train, Y_train = X_train[:(int(len(X_train)*train_size)),:], Y_train[:(int(len(X_train)*train_size)),:]
        result, train_pred, test_pred = svm_twostage(X_train, X_test, Y_train, Y_test, num_bus, kernel= 'poly',degree=2, epsilon = 0.01)
        R1.append(result)
    #    Process_pred = normalize(np.row_stack([train_pred, test_pred]), axis=0)
    #    train_pred, test_pred = Process_pred[:train_pred.shape[0], :], Process_pred[train_pred.shape[0]:, :]
        X_train_aug, X_test_aug = np.column_stack([X_train, train_pred]), np.column_stack([X_test, test_pred])
        result2, train_pred, test_pred = svm_twostage(X_train_aug, X_test_aug, Y_train, Y_test, num_bus, kernel= 'poly', degree=2, epsilon = 0.01)
        R2.append(result2)
    RR1.append(R1)
    RR2.append(R2)

print(np.mean(np.array(RR1), axis=0))
print(np.mean(np.array(RR2), axis=0))

print(np.round(np.mean(np.array(RR1), axis=0),3))
print(np.round(np.mean(np.array(RR2), axis=0),3))











## delete predictors

def inv_prepare_data(num_bus, X_std, Y_std, seed, test_size=0.1, miss= None):
    train_path = '/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + ' bus'

    V, Theta, P, Q, file_name = inv_load_data(train_path)
    np.random.seed(seed)
    P += np.random.normal(0, np.abs(np.mean(P, axis=0)) * X_std, (P.shape[0], P.shape[1]))
    Q += np.random.normal(0, np.abs(np.mean(Q, axis=0)) * X_std, (Q.shape[0], Q.shape[1]))

    V += np.random.normal(0, np.abs(np.mean(V, axis=0)) * Y_std, (V.shape[0], V.shape[1]))
    Theta += np.random.normal(0, np.abs(np.mean(Theta, axis=0)) * Y_std, (Theta.shape[0], Theta.shape[1]))
    X, Y = np.column_stack([P, Q]), np.column_stack([V, Theta, P, Q])
    if miss != None:
        if miss > 1 or miss < 0:
            print('adjust missing proportion!')
        else:
            Keep = np.sort(train_test_split(np.arange(X.shape[1]), test_size=int(X.shape[1] * miss))[0])
            X = X[:, np.sort(Keep)]
  #  X = X [: , miss].reshape(-1,1)
    train_index, test_index = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=seed)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    Y_bus, ref_bus = construct_Y_bus(num_bus)

    return X_train, X_test, Y_train, Y_test, Y_bus, ref_bus

def Penalty(PQ_true, V_theta_pred, num_bus):
    Voltage = nprect(V_theta_pred[:, :num_bus], V_theta_pred[:, num_bus:])
    formula = [np.multiply(Voltage[obs_idx, :], np.conj(np.matmul(Y_bus, Voltage.transpose()))[:, obs_idx]) for
               obs_idx in range(V_theta_pred.shape[0])]
    formula = np.array(formula)
    PQ_hat = np.column_stack([formula.real, formula.imag])
    return np.sqrt(mse(PQ_true, PQ_hat))
  #  return [np.sqrt(mse(PQ_true[:,i], PQ_hat[:,i])) for i in range(int(2*num_bus))]


RR1, RR2 = [], []   ## don't normalize X
for Seed in range(3):
    R1, R2= [], []
    for miss in ([None, 0.01, 0.05,0.1, 0.2 ]):
        X_std=0.01
        Y_std =X_std
        X_train, X_test, Y_train, Y_test, Y_bus, ref_bus= inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size, miss = miss)
        result, train_pred, test_pred = svm_twostage(X_train, X_test, Y_train, Y_test, num_bus, kernel= 'poly', degree=2, epsilon = 0.01)
        R1.append(result)
        X_train_aug, X_test_aug = np.column_stack([X_train, train_pred]), np.column_stack([X_test, test_pred])
        result2, train_pred, test_pred = svm_twostage(X_train_aug, X_test_aug, Y_train, Y_test, num_bus, kernel= 'poly',degree=2, epsilon = 0.01)
        R2.append(result2)
    RR1.append(R1)
    RR2.append(R2)

print(np.mean(np.array(RR1), axis=0))
print(np.mean(np.array(RR2), axis=0))

















### NN
def inv_single_NN(input_dim, num_neural1, lr=0.01):
    num_neural2 = 20
    inputs = Input(shape=(input_dim,), name='input')

    x = Dense(units=num_neural1, activation='relu', kernel_initializer=initializers.RandomNormal(seed=1),
              bias_initializer=initializers.Zeros())(inputs)

    y_prediction = Dense(units=1, activation=None, kernel_initializer=initializers.RandomNormal(seed=1),
                         bias_initializer=initializers.Zeros(), name='y1_predictions')(
        x)  # hiddens[bus_idx])

    model = Model(inputs=inputs, outputs=y_prediction)
    Adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=Adam, loss=inv_Regression_loss)
    return model

def inv_singleNN(idx, X_train, X_test, Y_train, Y_test, num_bus, lr,epoch, batch_size=64, val_split=0):
    y_train, y_test = Y_train[:,:(2*num_bus)], Y_test[:,:(2*num_bus)]
    inv_singleNN= inv_single_NN(X_train.shape[1], 40,lr)
    history= inv_singleNN.fit(X_train,y_train[:,idx],batch_size=batch_size, validation_split=val_split, epochs=epoch, verbose=2)
    train_pred = inv_singleNN.predict(X_train)
    train_rms = np.sqrt(mean_squared_error(y_train[:,idx], train_pred))

    test_pred = inv_singleNN.predict(X_test)
    test_rms = np.sqrt(mean_squared_error(y_test[:,idx], test_pred))

    return train_pred,  test_pred , train_rms, test_rms

def inv_singleNN_twostage( X_train, X_test, Y_train, Y_test, num_bus, lr, epoch):
    train_pred, test_pred, test_rms_INDI = [], [], []
    for idx in range(int(2*num_bus)):
        seed(1)
        tf.set_random_seed(1)
        train_pred_indi,  test_pred_indi , train_rms_indi, test_rms_indi = inv_singleNN(idx, X_train, X_test,
                        Y_train, Y_test, num_bus, lr=lr, epoch = epoch )
        train_pred.append(train_pred_indi), test_pred.append(test_pred_indi), test_rms_INDI.append(test_rms_indi)
    train_pred=np.column_stack(train_pred)
    test_pred=np.column_stack(test_pred)
    test_rms_INDI = np.column_stack(test_rms_INDI)

    train_rms = np.sqrt(mean_squared_error(Y_train[:,:(2*num_bus)], train_pred))
    train_pe_rms = Penalty(Y_train[:, (2 * num_bus):], train_pred, num_bus)
    test_rms = np.sqrt(mean_squared_error(Y_test[:,:(2*num_bus)], test_pred))
    test_pe_rms = Penalty(Y_test[:, (2 * num_bus):], test_pred, num_bus)
    result = np.array([train_rms, train_pe_rms, test_rms, test_pe_rms])

    return result, train_pred, test_pred, test_rms_INDI

def inv_NN_multistage(X_train, X_test, y_train, y_test, num_bus, val_split=0, batch_size=100, epoch=100):
    seed(1)
    tf.set_random_seed(1)
    d = inv_twostage(X_train.shape[1], num_bus, 40, 0.2)
  #  sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9)
    Adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    adam_callbacks = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=3, min_delta=0.),
                      ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                                        min_delta=1e-8, cooldown=0, min_lr=0)]

    d.compile(optimizer=Adam, loss=inv_Regression_loss)
    history = d.fit(X_train, y_train, batch_size=batch_size, validation_split=val_split, epochs=epoch, verbose=2)  # ,callbacks=adam_callbacks)
    train_pred = d.predict(X_train)
    train_rms = np.sqrt(mean_squared_error(y_train[:, :(2 * num_bus)], train_pred))
    train_pe_rms = Penalty(y_train[:, (2 * num_bus):], train_pred, num_bus)
    test_pred = d.predict(X_test)
    test_rms = np.sqrt(mean_squared_error(y_test[:, :(2 * num_bus)], test_pred))
    test_pe_rms = Penalty(y_test[:, (2 * num_bus):], test_pred, num_bus)

    result = np.array([train_rms, train_pe_rms, test_rms, test_pe_rms])

    loss = np.array(history.history['loss'])
    if val_split != 0:
        val_loss = np.array(history.history['val_loss'])
    else:
        val_loss = []

    return result, loss, val_loss, train_pred, test_pred



# NN single output
r1comb, r2comb = [], []
for Seed in range(3):
    num_bus, test_size = 14, 0.1
    R1, R2 = [], []
    for X_std in ([0]):
        Y_std = X_std
        X_train, X_test, Y_train, Y_test, Y_bus, ref_bus = inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size)
        result1, train_pred, test_pred, test_rms_INDI = inv_singleNN_twostage( X_train, X_test,
                                Y_train, Y_test, num_bus, lr=0.01, epoch = 50)
        R1.append(result1)
        Process_pred = normalize(np.row_stack([train_pred, test_pred]), axis=0)
        train_pred, test_pred = Process_pred[:train_pred.shape[0], :], Process_pred[train_pred.shape[0]:, :]

        X_train_aug, X_test_aug = np.column_stack([X_train, train_pred]), np.column_stack([X_test, test_pred])

        result2, train_pred, test_pred, test_rms_INDI = inv_singleNN_twostage( X_train_aug, X_test_aug,
                                Y_train, Y_test, num_bus, lr=0.01, epoch = 50)
        R2.append(result2)
    r1comb.append(R1)
    r2comb.append(R2)
print(np.mean(np.array(r1comb), axis=0))
print(np.mean(np.array(r2comb), axis=0))




# NN
r1comb, r2comb = [], []
for Seed in range(4):
    num_bus, test_size = 39, 0.1
    R1, R2 = [], []
    for X_std in ([0, 0.01]):
        Y_std = X_std
        X_train, X_test, Y_train, Y_test, Y_bus, ref_bus = inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size)
        result1, loss, val_loss, train_pred, test_pred = inv_NN_multistage(X_train, X_test, Y_train, Y_test, num_bus)
        R1.append(result1)
        Process_pred = normalize(np.row_stack([train_pred, test_pred]), axis=0)
        train_pred, test_pred = Process_pred[:train_pred.shape[0], :], Process_pred[train_pred.shape[0]:, :]

        X_train_aug, X_test_aug = np.column_stack([X_train, train_pred]), np.column_stack([X_test, test_pred])

        result2, loss, val_loss, train_pred, test_pred = inv_NN_multistage(X_train_aug, X_test_aug, Y_train, Y_test,
                                                                           num_bus)
        R2.append(result2)
    r1comb.append(R1)
    r2comb.append(R2)
print(np.mean(np.array(r1comb), axis=0))
print(np.mean(np.array(r2comb), axis=0))
