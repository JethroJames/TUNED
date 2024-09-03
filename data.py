import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = data_X.shape[0]
        print(self.num_views)
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5,
                       views_to_add=None):
        if addNoise:
            # self.addNoise(index, ratio_noise, sigma=sigma)
            self.addNoise(index, ratio_noise, sigma=sigma, views_to_noise=views_to_add)
        if addConflict:
            self.addConflict(index, ratio_conflict, views_to_conflict=views_to_add)
        pass

    def addNoise(self, index, ratio, sigma, views_to_noise):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        if views_to_noise != None:
            for i in selects:
                views = np.array(views_to_noise) if views_to_noise is not None else np.arange(self.num_views)
                for v in views:
                    self.X[v][i] += np.random.normal(0, sigma, size=self.X[v][i].shape)
        else:
            for i in selects:
                views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views),
                                         replace=False)
                for v in views:
                    self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio, views_to_conflict):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        if views_to_conflict != None:
            for i in selects:
                views = np.array(views_to_conflict) if views_to_conflict is not None else np.arange(self.num_views)
                for v in views:
                    conflict_class = (self.Y[i] + 1) % self.num_classes
                    self.X[v][i] = records[conflict_class][v]
        else:
            for i in selects:
                v = np.random.randint(self.num_views)
                self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass


def PIE():
    data_path = "data/PIE_face_10.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("PIE", data_X, data_Y)


def HandWritten():
    data_path = "data/handwritten.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("HandWritten", data_X, data_Y)


def Scene():
    data_path = "data/scene15_mtv.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("Scene", data_X, data_Y)

if __name__ == '__main__':
    dataset3 = PIE()
    print(dataset3)
