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
                                                                                                           axis=0), np.stack(
        VI_Q, axis=0)
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


def make_dragonnet(input_dim, num_neural1, dropout):
    num_neural2 = 20
    inputs = Input(shape=(input_dim,), name='input')

    x = Dense(units=num_neural1, activation='relu', kernel_initializer='RandomNormal',
              bias_initializer=initializers.Zeros())(inputs)
    #              x = Dropout(dropout)(x)
    #               x = Dense(units=num_neural1, activation='relu', kernel_initializer='RandomNormal',
    #                    bias_initializer=initializers.Zeros())(x)
    #            x = Dropout(dropout)(x)

    #          hiddens=[]
    #          for bus_idx in range(0, int(input_dim * 0.5)):
    #              hidden = Dense(units=num_neural2,  activation='relu', kernel_initializer='RandomNormal',
    #                 bias_initializer=initializers.Zeros(), name='hidden' + str(bus_idx))(x)
    #              hidden = Dropout(dropout)(hidden)
    #              hiddens.append(hidden)

    y_predictions = []
    for bus_idx in range(0, int(input_dim * 0.5)):
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

def prepare_data(num_bus, err_std, seed, test_size):
        train_path = '/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + ' bus'
        # num_bus = int(re.compile(r'\d+').findall(train_path)[-1])
        X, Y_P, VI_P, Y_Q, VI_Q, file_name = load_data(train_path)
        outlier = np.unique(np.where(np.abs(Y_P - VI_P)[:, ] > 1)[0])
        X, Y_P, VI_P = np.delete(X, outlier, 0), np.delete(Y_P, outlier, 0), np.delete(VI_P, outlier, 0)

        X = X + np.random.normal(0, err_std, (X.shape[1], X.shape[0])).transpose()

        real = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/real_Y_bus.txt', dtype='float64',
                          delimiter=',')
        imag = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/imag_Y_bus.txt', dtype='float64',
                          delimiter=',')
        ref_bus = np.loadtxt('/Users/jiahe/Desktop/dragon/data/' + str(num_bus) + '/ref_bus.txt', dtype='int32',
                             delimiter=',')

        Y_bus = real + 1j * imag
        Voltage = nprect(X[:, :num_bus], X[:, num_bus:])
        formula = [np.multiply(Voltage[obs_idx, :], np.conj(np.matmul(Y_bus, Voltage.transpose()))[:, obs_idx]) for
                   obs_idx in range(X.shape[0])]
        formula = np.array(formula)
        VI_P, VI_Q = formula.real, formula.imag
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

        return X_train, X_test, P_train, P_test, Q_train, Q_test, np.mean(np.abs(Y_P - VI_P)), np.mean(np.abs(Y_Q - VI_Q))






