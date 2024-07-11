import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SOSREP.src.sosrep import SOSREP
from SOSREP.utils.DAL import load_from_ADbench



all_sets = ['7_Cardiotocography','24_mnist', '11_donors', '36_speech', '46_WPBC', '18_Ionosphere', '13_fraud', '20_letter',
                '32_shuttle', '22_magic.gamma',
                '35_SpamBase', '43_WDBC', '10_cover', '19_landsat', '14_glass', '23_mammography', '26_optdigits',
                '31_satimage-2', '30_satellite', '42_WBC',
                '3_backdoor', '27_PageBlocks', '47_yeast', '21_Lymphography', '41_Waveform', '44_Wilt', '2_annthyroid',
                '37_Stamps', '38_thyroid', '39_vertebral',
                '8_celeba', '28_pendigits', '9_census', '25_musk', '34_smtp', '29_Pima',
                '15_Hepatitis', '45_wine', '33_skin', '6_cardio'  ,'1_ALOI', '17_InternetAds', '40_vowels', '4_breastw', '16_http', '5_campaign', '12_fault']


all_aucs = []

for data_set in all_sets:
    data_dict  = load_from_ADbench(data_set)
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y = data_dict['Y_test']

    sosrep = SOSREP()
    _, _, fds, _ = sosrep.fit_predict(X_train, X_test, kernel_type='Laplacian')
    f = sosrep.predict(X_test)

    auc = sklearn.metrics.roc_auc_score(y,- f.cpu().detach().numpy())
    all_aucs.append(auc)


    plt.plot(fds)
    plt.show()


print(all_aucs)