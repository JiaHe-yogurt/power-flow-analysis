def load_data(train_path):
    X, Y_P, VI_P, Y_Q, VI_Q = [], [], [], [], []
    file_name = []
    print('Going to read images')
    path = os.path.join(train_path, '*txt')
    files = glob.glob(path)
    for fl in files:
        dat = np.loadtxt(fl, dtype='float64', delimiter=',')
        x = dat[:, :2]
        y_P, vi_P = dat[:, 2], dat[:, 4]
        y_Q, vi_Q = dat[:, 3], dat[:, 5]
        y_P, vi_P = np.reshape(y_P.transpose(), -1), np.reshape(vi_P.transpose(), -1)
        y_Q, vi_Q = np.reshape(y_Q.transpose(), -1), np.reshape(vi_Q.transpose(), -1)
        x = np.reshape(x.transpose(), -1)
        X.append(x), Y_P.append(y_P), VI_P.append(vi_P), Y_Q.append(y_Q), VI_Q.append(vi_Q)
        flbase = os.path.basename(fl)
        file_name.append(flbase)
    X, Y_P, VI_P, Y_Q, VI_Q = np.stack(X, axis=0), np.stack(Y_P, axis=0), np.stack(VI_P, axis=0), np.stack(Y_Q,
                            axis=0), np.stack(VI_Q, axis=0)
    file_name = np.stack(file_name, axis=0)

    return X, Y_P, VI_P, Y_Q, VI_Q, file_name


def regularize_loss(ratio):
    def regression_loss(concat_true, concat_pred):
        y_true = [concat_true[:, k] for k in range(0, concat_pred.shape[1])]
        v_true = [concat_true[:, k] for k in range(concat_pred.shape[1], 2 * concat_pred.shape[1])]
        y_pred = [concat_pred[:, k] for k in range(0, concat_pred.shape[1])]

        loss = [
            tf.reduce_mean(tf.square(y_true[k] - y_pred[k])) + ratio * tf.reduce_mean(tf.square(v_true[k] - y_pred[k]))
            for k in range(len(y_true))]

        return tf.reduce_mean(loss)

    return regression_loss


def regression_loss(concat_true, concat_pred):
    y_true = [concat_true[:, k] for k in range(0, concat_pred.shape[1])]
    y_pred = [concat_pred[:, k] for k in range(0, concat_pred.shape[1])]

    loss = [tf.reduce_mean(tf.square(y_true[k] - y_pred[k]))
            for k in range(len(y_true))]

    return tf.reduce_mean(loss)


def penalty_loss(concat_true, concat_pred):
    v_true = [concat_true[:, k] for k in range(concat_pred.shape[1], 2 * concat_pred.shape[1])]
    y_pred = [concat_pred[:, k] for k in range(0, concat_pred.shape[1])]
    loss = [tf.reduce_mean(tf.square(v_true[k] - y_pred[k])) for k in range(len(v_true))]

    return tf.reduce_mean(loss)


def make_dragonnet(num_bus, input_dim, num_neural1=40, dropout=0.2):
    num_neural2 = 20
    inputs = Input(shape=(input_dim,), name='input')

    x = Dense(units=num_neural1, activation='relu', kernel_initializer='RandomNormal',
              bias_initializer=initializers.Zeros())(inputs)
    x = Dropout(dropout)(x)


    y_predictions = []
    for bus_idx in range(0, num_bus):
        y_prediction = Dense(units=1, activation=None, kernel_initializer='RandomNormal',
                             bias_initializer=initializers.Zeros(), name='y' + str(bus_idx) + '_predictions')(x)
        y_predictions.append(y_prediction)

    concat_pred = Concatenate(1)(y_predictions)

    model = Model(inputs=inputs, outputs=concat_pred)

    return model


def single_make_dragonnet(input_dim, num_neural1, dropout):
    num_neural2 = 20
    inputs = Input(shape=(input_dim,), name='input')

    x = Dense(units=num_neural1, activation='relu', kernel_initializer='RandomNormal',
              bias_initializer=initializers.Zeros())(inputs)
    x = Dropout(dropout)(x)
    x = Dense(units=num_neural1, activation='relu', kernel_initializer='RandomNormal',
              bias_initializer=initializers.Zeros())(x)
    x = Dropout(dropout)(x)

    #          hiddens=[]
    y_prediction = Dense(units=1, activation=None, kernel_initializer='RandomNormal',
                         bias_initializer=initializers.Zeros(), name='y1_predictions')(
        x)  # hiddens[bus_idx])

    model = Model(inputs=inputs, outputs=y_prediction)

    return model

def prepare_data(num_bus, X_std, Y_std, seed, test_size=0.1):
        train_path = '/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + ' bus'
        # num_bus = int(re.compile(r'\d+').findall(train_path)[-1])
        X, Y_P, VI_P, Y_Q, VI_Q, file_name = load_data(train_path)

        X += np.random.normal(0, np.abs(np.mean(X, axis=0)) * X_std, (X.shape[0], X.shape[1]))
        Y_P += np.random.normal(0, np.abs(np.mean(Y_P, axis=0)) * Y_std, (Y_P.shape[0], Y_P.shape[1]))
        Y_Q += np.random.normal(0, np.abs(np.mean(Y_Q, axis=0)) * Y_std, (Y_Q.shape[0], Y_Q.shape[1]))

        real = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/real_Y_bus.txt', dtype='float64',
                          delimiter=',')
        imag = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/imag_Y_bus.txt', dtype='float64',
                          delimiter=',')
        ref_bus = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/ref_bus.txt', dtype='int32',
                             delimiter=',')

        Y_bus = real + 1j * imag
        nprect = np.vectorize(rect)
        Voltage = nprect(X[:, :num_bus], X[:, num_bus:])
        formula = [np.multiply(Voltage[obs_idx, :], np.conj(np.matmul(Y_bus, Voltage.transpose()))[:, obs_idx]) for
                   obs_idx in range(X.shape[0])]
        formula = np.array(formula)
        VI_P, VI_Q = formula.real, formula.imag
        VI_Q, Y_Q= normalize(VI_Q), normalize(Y_Q)
        VI_P, Y_P= normalize(VI_P), normalize(Y_P)

        theta = X[:, num_bus:] * 180 / np.pi
        theta[:, ref_bus - 1] = 45
        V = X[:, :num_bus]
        u, v = np.multiply(V, np.cos(theta)), np.multiply(V, np.sin(theta))
        X = np.column_stack([u, v])
        # print(np.mean(np.abs(Y_P[:, bus_idx] - VI_P[:, bus_idx])))
        tf.set_random_seed(seed)
        np.random.seed(seed)
        train_index, test_index = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=seed)
        X_train, X_test = X[train_index], X[test_index]
        Y_P_train, Y_P_test, VI_P_train, VI_P_test = Y_P[train_index], Y_P[test_index], VI_P[train_index], VI_P[
            test_index]
        Y_Q_train, Y_Q_test, VI_Q_train, VI_Q_test = Y_Q[train_index], Y_Q[test_index], VI_Q[train_index], VI_Q[
            test_index]

        P_train = np.column_stack([Y_P_train, VI_P_train])
        Q_train = np.column_stack([Y_Q_train, VI_Q_train])
        P_test = np.column_stack([Y_P_test, VI_P_test])
        Q_test = np.column_stack([Y_Q_test, VI_Q_test])

        return X_train, X_test, P_train, P_test, Q_train, Q_test, np.mean(np.divide(VI_P-Y_P,Y_P),axis=0), np.mean(np.divide(VI_Q-Y_Q,Y_Q),axis=0)


        epoch = 200
        batch_size = 128
        val_split = 0.1
        nprect = np.vectorize(rect)

        # NN multiple output

def score(concat_true, concat_pred):
        y_true = [concat_true[:, k] for k in range(0, concat_pred.shape[1])]
        v_true = [concat_true[:, k] for k in range(concat_pred.shape[1], 2 * concat_pred.shape[1])]
        y_pred = [concat_pred[:, k] for k in range(0, concat_pred.shape[1])]
        loss = [np.mean(np.square(y_true[k] - y_pred[k]))
                + np.mean(np.square(v_true[k] - y_pred[k]))  for k in range(len(y_true))]
        return np.mean(loss)

def NN(X_train, X_test,y_train,y_test,Ratio,num_bus,val_split=0,batch_size=200,epoch=200):
            result = []
            Test_pred =[]
            loss, val_loss, reg_loss, val_reg_loss, pe_loss, val_pe_loss = [], [], [], [], [], []
            for ratio in Ratio:
                d = make_dragonnet(num_bus, X_train.shape[1], 40, 0.2)
                #    sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9)
                Adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
                metrics = [regression_loss, penalty_loss]
                adam_callbacks = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=10, min_delta=0.),
                                  ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                                                    min_delta=1e-8, cooldown=0, min_lr=0)]

                d.compile(optimizer=Adam, loss=regularize_loss(ratio=ratio), metrics=metrics)
                history = d.fit(X_train, y_train, batch_size=batch_size, validation_split=val_split, epochs=epoch,
                                verbose=2)#, callbacks=adam_callbacks)
                train_pred = d.predict(X_train)
                train_rms= np.sqrt((mean_squared_error(y_train[:, :num_bus], train_pred)))
                train_vi_rms = np.sqrt((mean_squared_error(y_train[:, num_bus:], train_pred)))

                test_pred = d.predict(X_test)
                Test_pred.append(test_pred)
                test_rms = np.sqrt((mean_squared_error(y_test[:, :num_bus], test_pred)))
                test_vi_rms = np.sqrt((mean_squared_error(y_test[:, num_bus:], test_pred)))

                result.append([train_rms, train_vi_rms, test_rms, test_vi_rms])

                loss.append(history.history['loss'])
                reg_loss.append(history.history['regression_loss'])
                pe_loss.append(history.history['penalty_loss'])

                if val_split != 0:
                    val_loss.append(history.history['val_loss'])
                    val_reg_loss.append(history.history['val_regression_loss'])
                    val_pe_loss.append(history.history['val_penalty_loss'])
                else:
                    val_loss, val_reg_loss, val_pe_loss = [], [], []

            result = np.array(result)
            loss = np.array(loss)
            val_loss = np.array(val_loss),
            val_reg_loss = np.array(val_reg_loss)
            val_pe_loss = np.array(val_pe_loss)
            reg_loss = np.array(reg_loss)
            pe_loss = np.array(pe_loss)

            return result, loss, reg_loss, pe_loss,  val_loss, val_reg_loss, val_pe_loss,Test_pred

    num_bus, X_std,Y_std, seed, test_size = 9, 0.005, 0.1, 1,0.1
    Ratio = ([0,1,10])
    X_train, X_test, P_train, P_test, Q_train, Q_test, P_diff, Q_diff=prepare_data(num_bus, X_std, Y_std, seed, test_size)

    result, loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss,Test_pred= NN(X_train, X_test, Q_train, Q_test, Ratio,num_bus , epoch=200)



for:
        i = 0
        plt.subplot(241)
        plt.plot(reg_loss[i], label='regression loss')
        plt.plot(pe_loss[i], label='penalty loss')
        plt.title('lambda=0', {'fontsize': 10})
        plt.legend()
        plt.subplot(245)
        plt.plot(val_reg_loss[i])
        plt.plot(val_pe_loss[i])
        plt.title('lambda=0', {'fontsize': 10})

        i = 1
        plt.subplot(242)
        plt.plot(reg_loss[i], label='regression loss')
        plt.plot(pe_loss[i], label='penalty loss')
        plt.title('lambda=0.1', {'fontsize': 10})

        plt.subplot(246)
        plt.plot(val_reg_loss[i])
        plt.plot(val_pe_loss[i])
        plt.title('lambda=0.1', {'fontsize': 10})

        i = 2
        plt.subplot(243)
        plt.plot(reg_loss[i], label='regression loss')
        plt.plot(pe_loss[i], label='penalty loss')
        plt.title('lambda=1', {'fontsize': 10})

        plt.subplot(247)
        plt.plot(val_reg_loss[i])
        plt.plot(val_pe_loss[i])
        plt.title('lambda=1', {'fontsize': 10})

        i = 3
        plt.subplot(244)
        plt.plot(reg_loss[i], label='regression loss')
        plt.plot(pe_loss[i], label='penalty loss')
        plt.title('lambda=10', {'fontsize': 10})

        plt.subplot(248)
        plt.plot(val_reg_loss[i])
        plt.plot(val_pe_loss[i])
        plt.title('lambda=10', {'fontsize': 10})


def CV_make_dragonnet(ratio, input_dim=2*num_bus, num_neural1=40, dropout=0.2):
    num_neural2 = 20

    inputs = Input(shape=(input_dim,), name='input')

    x = Dense(units=num_neural1, activation='relu', kernel_initializer='RandomNormal',
              bias_initializer=initializers.Zeros())(inputs)
    x = Dropout(dropout)(x)

    y_predictions = []
    for bus_idx in range(0, num_bus):
        y_prediction = Dense(units=1, activation=None, kernel_initializer='RandomNormal',
                             bias_initializer=initializers.Zeros(), name='y' + str(bus_idx) + '_predictions')(
            x)  # hiddens[bus_idx])
        y_predictions.append(y_prediction)

    concat_pred = Concatenate(1)(y_predictions)

    model = Model(inputs=inputs, outputs=concat_pred)

    Adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    metrics = [regression_loss, penalty_loss]

    model.compile(optimizer=Adam, loss=regularize_loss(ratio=ratio), metrics=metrics)
    return model


def CV_NN(X_train, X_test, y_train, y_test, ratio, num_bus, val_split=0, batch_size=200, epoch=200):
    d = make_dragonnet(num_bus, X_train.shape[1], 40, 0.2)
    #    sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9)
    Adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    metrics = [regression_loss, penalty_loss]
    adam_callbacks = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=3, min_delta=0.),
                      ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                                        min_delta=1e-8, cooldown=0, min_lr=0)]

    d.compile(optimizer=Adam, loss=regularize_loss(ratio=ratio), metrics=metrics)
    history = d.fit(X_train, y_train, batch_size=batch_size, validation_split=val_split, epochs=epoch,
                    verbose=2)  # ,callbacks=adam_callbacks)
    train_pred = d.predict(X_train)
    mse = mean_squared_error
    train_rms = np.sqrt((mse(y_train[:, :num_bus], train_pred)))
    train_vi_rms = np.sqrt((mse(y_train[:, num_bus:], train_pred)))

    test_pred = d.predict(X_test)
    test_rms = np.sqrt((mse(y_test[:, :num_bus], test_pred)))
    test_vi_rms = np.sqrt((mse(y_test[:, num_bus:], test_pred)))

    result = [train_rms, train_vi_rms, test_rms, test_vi_rms]

    loss = np.array(history.history['loss'])
    reg_loss = np.array(history.history['regression_loss'])
    pe_loss = np.array(history.history['penalty_loss'])
    if val_split != 0:
        val_loss = np.array(history.history['val_loss'])
        val_reg_loss = np.array(history.history['val_regression_loss'])
        val_pe_loss = np.array(history.history['val_penalty_loss'])
    else:
        val_loss, val_reg_loss, val_pe_loss = [], [], []
    return result, loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss

    ## forward


seed = 1
np.random.seed(seed)
batch_size = 200
epochs = 200

#num_bus, X_std, Y_std, seed, test_size = 9, 0.005, 0.1, 1, 0.1
num_bus, X_std, Y_std, seed, test_size = 14, 0.05, 0.1, 1, 0.1  #real power
X_train, X_test, P_train, P_test, Q_train, Q_test, P_diff, Q_diff = prepare_data(num_bus, X_std, Y_std, seed, test_size)
Y_bus, ref_bus = construct_Y_bus(num_bus)

np.random.seed(seed)
model_CV = KerasRegressor(build_fn=CV_make_dragonnet, nb_epoch=200, batch_size=200, verbose=0)
# define the grid search parameters

init_mode = np.arange(0,10,1)

param_grid = dict(ratio=init_mode)

# define score () to select model
scoring = {'loss': make_scorer(score, greater_is_better=False)}
grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3, scoring=scoring, refit='loss')
Y_train, Y_test = P_train, P_test
grid_result = grid.fit(X_train, Y_train)

best_para = grid_result.best_params_.get('ratio')

#plt.plot(np.array(np.abs(grid_result.cv_results_.get('mean_test_loss'))), marker='+')

#result, loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss = CV_NN(X_train, X_test, Y_train, Y_test, best_para, num_bus)
Ratio=([0,np.round(best_para,2),10])
result, loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss, Test_pred = NN(X_train, X_test,
                                                    Y_train, Y_test, Ratio, num_bus, epoch=150)


for i in range(len(Ratio)):
    plt.plot(pe_loss[i], label='$\lambda$=' + str(Ratio[i]), linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Penalty Loss')
plt.legend()


for i in range(len(Ratio)):
    plt.plot(reg_loss[i], label='$\lambda$=' + str(Ratio[i]), linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Regression Loss')
plt.legend()


x = np.linspace(0, 200, num=200, endpoint=True)















    R=[]     # NN single output
    yt_train, y_test = Q_train, Q_test
    for bus_idx in range(num_bus):
        result = []
        loss, val_loss, reg_loss, val_reg_loss, pe_loss, val_pe_loss = [], [], [], [], [], []
        print('bus ' +str(bus_idx))
        for ratio in ([0,0.5,1,10]):
                print('ratio is '+str(ratio))
           # for ratio in ([0, 0.01, 0.1, 1, 10]):
                d = single_make_dragonnet(X_train.shape[1], 40, 0.2)
                #   sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9)
                Adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
                metrics = [regression_loss, penalty_loss]
                adam_callbacks = [TerminateOnNaN(), EarlyStopping(monitor='val_loss', patience=3, min_delta=0.),
                                  ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                                                    min_delta=1e-8, cooldown=0, min_lr=0)]

                d.compile(optimizer=Adam, loss=regularize_loss(ratio=ratio), metrics=metrics)
                tmptrain = np.column_stack([yt_train[:, bus_idx], yt_train[:, bus_idx + num_bus]])
                tmptest = np.column_stack([y_test[:, bus_idx], y_test[:, bus_idx + num_bus]])
                history = d.fit(X_train, tmptrain, batch_size=batch_size, validation_split=val_split, epochs=epoch,
                                verbose=2)#, callbacks=adam_callbacks)
                train_pred = d.predict(X_train)
                train_rms =(mean_squared_error(tmptrain[:, 0], train_pred))
                train_vi_rms = (mean_squared_error(tmptrain[:, 1], train_pred))

                test_pred = d.predict(X_test)
                test_vi_rms = (mean_squared_error(tmptest[:, 1], test_pred))
                test_rms = (mean_squared_error(tmptest[:, 0], test_pred))
                result.append([train_rms, train_vi_rms, test_rms, test_vi_rms])
                np.array(result)
      #  R.append(np.array(result))
                loss.append(history.history['loss'])
                val_loss.append(history.history['val_loss'])
                reg_loss.append(history.history['regression_loss'])
                val_reg_loss.append(history.history['val_regression_loss'])
                pe_loss.append(history.history['penalty_loss'])
                val_pe_loss.append(history.history['val_penalty_loss'])
        result = np.array(result)
        R.append(result)
        loss = np.array(loss)
        val_loss = np.array(val_loss)
        reg_loss = np.array(reg_loss)
        val_reg_loss = np.array(val_reg_loss)
        pe_loss = np.array(pe_loss)
        val_pe_loss = np.array(val_pe_loss)





    # svm forward
    from sklearn.svm import SVR

        #    Y_train, Y_test = P_train, P_test
            Rtwo= []
            svmresult = []
            for bus_idx in range(num_bus):
                svr = SVR(kernel='rbf', C=1, epsilon=0.01).fit(X_train, Y_train[:, bus_idx])
                train_pred = svr.predict(X_train)
                svm_train_reg= np.sqrt(mse(Y_train[:, bus_idx], train_pred))
                svm_train_pe = np.sqrt(mse(Y_train[:, num_bus+bus_idx], train_pred))
                test_pred = svr.predict(X_test)
                svm_test_reg=np.sqrt(mse(Y_test[:, bus_idx], test_pred))
                svm_test_pe = np.sqrt(mse(Y_test[:, num_bus+bus_idx], test_pred))
                svmresult.append(np.array([svm_train_reg, svm_train_pe, svm_test_reg, svm_test_pe ]))
                Rtwo.append(r2_score(Y_test[:, bus_idx], test_pred))
            print(np.round(np.mean(svmresult,axis=0),3))
            print(np.round(result[1,:],3))
