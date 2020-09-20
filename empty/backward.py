nprect = np.vectorize(rect)


def inv_load_data(train_path):
    V, Theta, P, Q, = [], [], [], []
    file_name = []
    print('Going to read images')
    path = os.path.join(train_path, '*txt')
    files = glob.glob(path)
    for fl in files:
        dat = np.loadtxt(fl, dtype='float64', delimiter=',')
        v, theta, p, q = dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3]
        v, theta = np.reshape(v.transpose(), -1), np.reshape(theta.transpose(), -1)
        V.append(v), Theta.append(theta), P.append(p), Q.append(q)
        flbase = os.path.basename(fl)
        file_name.append(flbase)
    V, Theta, P, Q = np.stack(V, axis=0), np.stack(Theta, axis=0), np.stack(P, axis=0), np.stack(
        Q, axis=0)
    file_name = np.stack(file_name, axis=0)
    return V, Theta, P, Q, file_name


def inv_make_dragonnet(input_dim, num_neural1=40, dropout=0.2):
    inputs = Input(shape=(input_dim,), name='input')

    x = Dense(units=num_neural1, activation='relu', kernel_initializer=initializers.RandomNormal(seed=1),
              bias_initializer=initializers.Zeros())(inputs)

    y_predictions = []
    for bus_idx in range(0, int(input_dim)):
        y_prediction = Dense(units=1, activation=None, kernel_initializer=initializers.RandomNormal(seed=1),
                             bias_initializer=initializers.Zeros(), name='y' + str(bus_idx) + '_predictions')(x)
        y_predictions.append(y_prediction)

    concat_pred = Concatenate(1)(y_predictions)

    model = Model(inputs=inputs, outputs=concat_pred)

    return model


def inv_Regression_loss(concat_true, concat_pred):
    v_theta_true = concat_true[:, :(2 * num_bus)]
    return tf.reduce_mean(tf.square(v_theta_true - concat_pred))


def inv_Penalty_loss(concat_true, concat_pred):
    v, theta = concat_pred[:, :num_bus], concat_pred[:, num_bus:]
    x, y = tf.multiply(v, tf.cos(theta)), tf.multiply(v, tf.sin(theta))
    Voltage = tf.cast(tf.complex(x, y), tf.complex128)
    formula = tf.multiply(Voltage, tf.transpose(tf.conj(tf.matmul(Y_bus, tf.transpose(Voltage)))))
    PQ_hat = tf.concat([tf.real(formula), tf.imag(formula)], axis=1)
    PQ_true = concat_true[:, (2 * num_bus):]
    PQ_hat, PQ_true = tf.cast(PQ_hat, tf.float64), tf.cast(PQ_true, tf.float64)
    return tf.reduce_mean(tf.square(PQ_true - PQ_hat))


def inv_regularize_loss(ratio):
    def loss(concat_true, concat_pred):
        reg_loss = tf.cast(inv_Regression_loss(concat_true, concat_pred), tf.float64)
        pena_loss = tf.cast(ratio * inv_Penalty_loss(concat_true, concat_pred), tf.float64)
        return reg_loss + pena_loss

    return loss


def Penalty(PQ_true, V_theta_pred, num_bus):
    Voltage = nprect(V_theta_pred[:, :num_bus], V_theta_pred[:, num_bus:])
    formula = [np.multiply(Voltage[obs_idx, :], np.conj(np.matmul(Y_bus, Voltage.transpose()))[:, obs_idx]) for
               obs_idx in range(V_theta_pred.shape[0])]
    formula = np.array(formula)
    PQ_hat = np.column_stack([formula.real, formula.imag])
    return np.sqrt(mse(PQ_true, PQ_hat))


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
    X = normalize(X, axis=0)  # ,normalize(Q, axis=0)  minmax_scale gives worse reesult
 #   Y = normalize(Y, axis=0)
    train_index, test_index = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=seed)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    Y_bus, ref_bus = construct_Y_bus(num_bus)

    return X_train, X_test, Y_train, Y_test, Y_bus, ref_bus


def construct_Y_bus(num_bus):
    real = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/real_Y_bus.txt', dtype='float64',
                      delimiter=',')
    imag = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/imag_Y_bus.txt', dtype='float64',
                      delimiter=',')
    ref_bus = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/ref_bus.txt', dtype='int32',
                         delimiter=',')
    Y_bus = real + 1j * imag
    return Y_bus, ref_bus


def backward_score(concat_true, concat_pred):
        v_theta_true = concat_true[:, :(2 * num_bus)]
        reg_loss = np.mean(np.square(v_theta_true - concat_pred))
        v, theta = concat_pred[:, :num_bus], concat_pred[:, num_bus:]
        Voltage = nprect(v, theta)
        formula = [np.multiply(Voltage[obs_idx, :], np.conj(np.matmul(Y_bus, Voltage.transpose()))[:, obs_idx]) for
                   obs_idx in range(v.shape[0])]
        formula = np.array(formula)
        PQ_hat = np.column_stack([formula.real, formula.imag])
        PQ_true = concat_true[:, (2 * num_bus):]

        pena_loss = np.mean(np.square(PQ_true - PQ_hat))
        return reg_loss + pena_loss


def inv_NN(X_train, X_test, Y_train, Y_test, Ratio, num_bus, lr=0.01,val_split=0, batch_size=64, epoch=200):
        result, sepa_reg_loss = [], []
        loss, val_loss, reg_loss, val_reg_loss, pe_loss, val_pe_loss = [], [], [], [], [], []
        for ratio in Ratio:
            seed(1)
            tf.set_random_seed(1)
            d = inv_make_dragonnet(X_train.shape[1], 40, 0.2)
            # sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9)
            Adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
            metrics = [inv_Regression_loss, inv_Penalty_loss]

            d.compile(optimizer=Adam, loss=inv_regularize_loss(ratio=ratio), metrics=metrics)
            history = d.fit(X_train, Y_train, batch_size=batch_size, validation_split=val_split, epochs=epoch,
                            verbose=2)  # ,callbacks=adam_callbacks)
            train_pred = d.predict(X_train)
            train_rms = np.sqrt(mse(Y_train[:, :(2 * num_bus)], train_pred))
            train_pe_rms = Penalty(Y_train[:, (2 * num_bus):], train_pred, num_bus)
            test_pred = d.predict(X_test)
            test_rms = np.sqrt(mse(Y_test[:, :(2 * num_bus)], test_pred))
            test_pe_rms = Penalty(Y_test[:, (2 * num_bus):], test_pred, num_bus)

            result.append([train_rms, train_pe_rms, test_rms, test_pe_rms])

            loss.append(history.history['loss'])
            reg_loss.append(history.history['inv_Regression_loss'])
            pe_loss.append(history.history['inv_Penalty_loss'])

            if val_split != 0:
                val_loss.append(history.history['val_loss'])
                val_reg_loss.append(history.history['val_inv_Regression_loss'])
                val_pe_loss.append(history.history['val_inv_Penalty_loss'])
            else:
                val_loss, val_reg_loss, val_pe_loss = [], [], []

        result = np.array(result)
        loss = np.array(loss)
        val_loss = np.array(val_loss),
        val_reg_loss = np.array(val_reg_loss)
        val_pe_loss = np.array(val_pe_loss)
        reg_loss = np.array(reg_loss)
        pe_loss = np.array(pe_loss)

        return result, loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss, train_pred


  ## forward

 Seed = 1
 np.random.seed(Seed)
 batch_size = 64
 epochs = 20
 num_bus, X_std, Y_std, test_size = 14, 0, 0, 0.1
 R, Para=[],[]
 for Seed in range(3):
    #num_bus, X_std, Y_std, seed, test_size = 39, 0, 0.1, 1, 0.1  work
    X_train, X_test, Y_train, Y_test, Y_bus, ref_bus = inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size)
    X_train, X_test, Y_train, Y_test = X_train[:100, :], X_test[:20, :], Y_train[:100, :], Y_test[:20, :]


    def inv_CV_make_dragonnet(ratio, input_dim=X_train.shape[1], num_neural1=40):
        seed(1)
        tf.set_random_seed(1)

        inputs = Input(shape=(input_dim,), name='input')

        x = Dense(units=num_neural1, activation='relu', kernel_initializer=initializers.RandomNormal(seed=1),
                  bias_initializer=initializers.Zeros())(inputs)

        y_predictions = []
        for bus_idx in range(0, int(input_dim)):
            y_prediction = Dense(units=1, activation=None, kernel_initializer=initializers.RandomNormal(seed=1),
                                 bias_initializer=initializers.Zeros(), name='y' + str(bus_idx) + '_predictions')(
                x)  # hiddens[bus_idx])
            y_predictions.append(y_prediction)

        concat_pred = Concatenate(1)(y_predictions)

        model = Model(inputs=inputs, outputs=concat_pred)

        Adam = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
        metrics = [inv_Regression_loss]  # , inv_Penalty_loss]

        model.compile(optimizer=Adam, loss=inv_regularize_loss(ratio=ratio), metrics=metrics)
        return model

    model_CV = KerasRegressor(build_fn=inv_CV_make_dragonnet, nb_epoch=200, batch_size=batch_size, verbose=0)
    # define the grid search parameters

    init_mode = np.arange(0, 1.5, 0.2)

    param_grid = dict(ratio=init_mode)

    # define score () to select model
    scoring = { 'loss': make_scorer(backward_score , greater_is_better=False) }
    grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3 ,scoring= scoring, refit= 'loss')
    grid_result = grid.fit(X_train, Y_train)

    best_para=grid_result.best_params_.get('ratio')
    Para.append(best_para)
#    plt.plot(np.array(np.abs(grid_result.cv_results_.get('mean_test_loss'))), marker= '+')

  #  result, loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss=inv_CV_NN(X_train, X_test, Y_train, Y_test,best_para, num_bus)

   # Ratio = ([0, np.round(best_para, 2)])
    Ratio = ([best_para])
    result,  loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss, train_pred = inv_NN(X_train,
                    X_test, Y_train, Y_test, Ratio, num_bus, lr=0.005, epoch=200)
    R.append(result)
 print(np.round(np.mean(R,0),3))
 print(np.mean(Para))



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



    R , para = [], []
    for Seed in range(4):
        for X_std in ([ 0]):
            Y_std = X_std
            X_train, X_test, Y_train, Y_test, Y_bus, ref_bus = inv_prepare_data(num_bus, X_std, Y_std, Seed, test_size)

            model_CV = KerasRegressor(build_fn=inv_CV_make_dragonnet, nb_epoch=150, batch_size=64, verbose=0)

            init_mode = np.arange(0.5, 1.5,0.1)

            param_grid = dict(ratio=init_mode)

            # define score () to select model
            scoring = {'loss': make_scorer(backward_score, greater_is_better=False)}
            grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3, scoring=scoring, refit='loss')
            grid_result = grid.fit(X_train, Y_train)

            best_para = grid_result.best_params_.get('ratio')
            Ratio = ([ np.round(best_para)])
            result, loss, reg_loss, pe_loss, val_loss, val_reg_loss, val_pe_loss, train_pred = inv_NN(X_train,
                                    X_test, Y_train, Y_test, Ratio, num_bus, epoch=150)
        R.append(result)
        para.append(best_para)
    np.round(np.mean(np.array(R),axis=0),3)


