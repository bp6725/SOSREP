from SOSREP.third_party.ADbench.data_generator import DataGenerator
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
class AdBench_Simulator():
    def __init__(self):
      pass

    @staticmethod
    def load_datasets(data_sets=None, params = {}):
        '''
        We return list of datasets. For each dataset we return 1.'X_train' 2.'X_test' 3. 'Y_test' 4. 'name' 5.'params'
        :param data_sets: datasets names in list
        :param params:
        :return:
        '''
        if data_sets is None:
            data_sets = ['2_annthyroid']

        all_sets = []
        for data_set_name in data_sets :
            set = AdBench_Simulator.load_dataset(data_set_name, params['la'])

            all_sets.append(set)

        return all_sets

    @staticmethod
    def load_dataset(data_set = '2_annthyroid', la = 0.1,resample = True, return_all_data = False):
        data_generator = DataGenerator(dataset = data_set,generate_duplicates=resample)
        data = data_generator.generator(la =la,return_all_data=return_all_data)
        return data



def load_from_ADbench(dataset_name, n_dims_to_take = -1, n_samples = -1, la = 0.1) :
    dataset = AdBench_Simulator.load_dataset(dataset_name, la,False,False)
    X_train, X_test = dataset['X_train'], dataset['X_test']
    Y_train, Y_test = dataset['y_train'], dataset['y_test']

    if n_samples != -1:
        X_train = X_train[0:n_samples]

    if n_dims_to_take != -1:
        raise NotImplementedError('No other dims for you.')

    #normliaze X_train and X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    running_results = {}
    running_results['X_train'] = X_train
    running_results['X_test'] = X_test
    running_results['Y_train'] = Y_train
    running_results['Y_test'] = Y_test
    running_results['dataset_name'] = dataset_name
    running_results["is supervised"] = sum(Y_train) > 0
    running_results["dim"] = X_train.shape[1]

    return running_results