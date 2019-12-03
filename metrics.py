import numpy as np


class Metrics(object):
    def __init__(self, y_true, y_pred, threshold=0.5):
        """y_true and y_pred should be same dims"""
        self.y_true = y_true
        self.y_pred = y_pred
        self.bool_y_true = (y_true == np.max(y_true))
        self.bool_y_pred = (y_pred > threshold)
        self.threshold = threshold
        self.TP, self.TN, self.FP, self.FN = self._cal_base()

    def get_dice_coef(self, smooth=0):
        # Dice coefficient(A,B) = 2*(A n B)/(|A| + |B|)
        dice_coefficient = (2 * self.TP + smooth) / (2 * self.TP + self.FP + self.FN + smooth)
        return dice_coefficient

    def get_iou(self):
        # IOU(A,B) = (A n B)/(A u B)
        iou = self.TP / (self.TP + self.FP + self.FN)
        return iou

    def _cal_base(self):
        TP = np.logical_and(self.bool_y_pred == True, self.bool_y_true == True)
        TN = np.logical_and(self.bool_y_pred == False, self.bool_y_true == False)
        FP = np.logical_and(self.bool_y_pred == True, self.bool_y_true == False)
        FN = np.logical_and(self.bool_y_pred == False, self.bool_y_true == True)

        TP = np.sum(TP)
        TN = np.sum(TN)
        FP = np.sum(FP)
        FN = np.sum(FN)
        return TP, TN, FP, FN

    def get_accuracy(self):
        _accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        return _accuracy

    def get_recall(self):
        _recall = self.TP / (self.TP + self.FN)
        return _recall

    def get_precision(self):
        _precision = self.TP / (self.TP + self.FP)
        return _precision

    def get_F1_score(self):
        _precision = self.get_precision()
        _recall = self.get_recall()
        _F1_score = 2 * (_precision * _recall) / (_precision + _recall + 1e-7)
        return _F1_score

    def get_specificity(self):
        _specificity = self.TN / (self.TN + self.FP)
        return _specificity

    def get_info(self):
        print("y_true:", self.y_true)
        print("y_pred:", self.y_pred)
        print("bool_y_true:", self.bool_y_true)
        print("bool_y_pred:", self.bool_y_pred)
        print("threshold:", self.threshold)


if __name__ == "__main__":
    y_pred = np.random.rand(5, 5)
    y_true = np.random.randint(0, 2, size=(5, 5))
    metrics = Metrics(y_true, y_pred, threshold=0.5)
    print(metrics.get_info())
    print(metrics._cal_base())
