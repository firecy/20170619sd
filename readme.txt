This is introduction of principal location of the faults-chain in wind turbine system.

functions document in preprocessing.py file
    1) get_limits(): return variables limits
    2) get_mean_std_us(xset, limits_array):
        input: xset, one year's wind dataset is better;
               limits_array, the limits got from "get_limits()".
        return: x_mean, x_std, x_u, x_s
    3) load_trainset(xset, limits_array, x_mean, x_std, x_u, x_s, epsilon):
        process 3D xset.
        input: xset, one year's wind dataset is better;
               limits_array, the limits got from "get_limits";
               x_mean, the mean value array got from "get_mean_std_us";
               x_std, the standard deviation value array got from "get_mean_std_us";
               x_u, the u value array got from "get_mean_std_us";
               x_s, the s value array got from "get_mean_std_us";
        return: xset_pre
    4) trainset_trans(trainset, labels):
        input: trainset, list/3D array
               labels, vector
        return: x0, 2D array
                y0, vector, length is equal to x0.shape[0]

functions document in sae.py file
    1)
