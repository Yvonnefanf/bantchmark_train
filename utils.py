"""create noise by sampling"""
import numpy as np

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y
#### label2 calss as same 
def confusion_noisyfy(y, P,random_state=0):
    # print(np.max(y), P.shape[0])
    m = y.shape[0]
    new_y = y.copy()
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    for idx in np.arange(m):
        if new_y[idx] == 2:
 
            new_y[idx] = 3
    return new_y

#### exchange 2 label
# def exchange_noisyfy(y, P,random_state=0):
#     # print(np.max(y), P.shape[0])
#     m = y.shape[0]
#     new_y = y.copy()
#     assert P.shape[0] == P.shape[1]
#     assert np.max(y) < P.shape[0]

#     for idx in np.arange(m):
#         if new_y[idx] == 2:
#             new_y[idx] = 3
#         elif new_y[idx] == 3:
#             new_y[idx] = 2
#     print('nee',new_y)
#     return new_y
        



def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = confusion_noisyfy(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else: 
        actual_noise = 0

    print(P)


    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
    print(P)
    return y_train_noisy, actual_noise


def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate