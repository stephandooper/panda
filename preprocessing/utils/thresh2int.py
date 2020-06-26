import copy
def thresh2int(thresholds, preds_raw):
    y_hat=copy.deepcopy(preds_raw)
    
    for i,pred in enumerate(y_hat):
        if   pred < thresholds[0]: y_hat[i] = 0
        elif pred < thresholds[1]: y_hat[i] = 1
        elif pred < thresholds[2]: y_hat[i] = 2
        elif pred < thresholds[3]: y_hat[i] = 3
        elif pred < thresholds[4]: y_hat[i] = 4
        else: y_hat[i] = 5
    return y_hat.astype('int')