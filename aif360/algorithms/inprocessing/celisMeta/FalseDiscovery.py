import numpy as np

from aif360.algorithms.inprocessing.celisMeta.General import General


class FalseDiscovery(General):
    def getExpectedGrad(self, dist, a, b, params, samples, mu, z_prior):
        t, probc_m1_0, probc_m1_1, prob_z_0, prob_z_1 = self.getValueForX(dist,
                a, b, params, z_prior, samples, return_probs=True)
        res = np.vstack([probc_m1_0 - a*prob_z_0,
                         probc_m1_1 - a*prob_z_1,
                        -probc_m1_0 + b*prob_z_0,
                        -probc_m1_1 + b*prob_z_1])
        res *= t / np.sqrt(t**2 + mu**2)
        return np.mean(res, axis=1)

    def getValueForX(self, dist, a, b, params, z_prior, x, return_probs=False):
        u_1, u_2, l_1, l_2 = params
        z_0, z_1 = 1-z_prior, z_prior

        pos = np.ones(len(x))
        try:
            prob_1_1 = self.prob(dist, np.c_[x, pos, pos])
            prob_m1_1 = self.prob(dist, np.c_[x, -pos, pos])
            prob_1_0 = self.prob(dist, np.c_[x, pos, np.zeros(len(x))])
            prob_m1_0 = self.prob(dist, np.c_[x, -pos, np.zeros(len(x))])
        except:
            prob_1_1 = self.prob(dist, np.c_[x, pos, pos, pos])
            prob_m1_1 = self.prob(dist, np.c_[x, -pos, pos, pos])
            prob_1_0 = self.prob(dist, np.c_[x, pos, np.zeros(len(x)), np.zeros(len(x))])
            prob_m1_0 = self.prob(dist, np.c_[x, -pos, np.zeros(len(x)), np.zeros(len(x))])

        total = prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1
        # if total == 0:
        #     return 0

        prob_y_1 = (prob_1_1 + prob_1_0) / total
        prob_z_0 = (prob_m1_0 + prob_1_0) / total
        prob_z_1 = (prob_m1_1 + prob_1_1) / total

        probc_m1_0 = prob_m1_0 / total
        probc_m1_1 = prob_m1_1 / total

        c_0 = prob_y_1 - 0.5
        c_1 = u_1*(probc_m1_0 - a*prob_z_0) + u_2*(probc_m1_1 - a*prob_z_1)
        c_2 = l_1*(-probc_m1_0 + b*prob_z_0) + l_2*(-probc_m1_1 + b*prob_z_1)

        t = c_0 + c_1 + c_2
        if return_probs:
            return t, probc_m1_0, probc_m1_1, prob_z_0, prob_z_1
        return t

    def getFuncValue(self, dist, a, b, params, samples, z_prior):
        return np.mean(np.abs(self.getValueForX(dist, a, b, params, z_prior,
                                                samples)))

    @property
    def num_params(self):
        return 4

    def gamma(self, y_true, y_pred, sens):
        try:
            pos_0 = y_pred[sens == 0] == 1
            pos_1 = y_pred[sens == 1] == 1
            if np.sum(pos_0) == 0 or np.sum(pos_1) == 0:
                return 0
            fdr_0 = np.sum(pos_0 & (y_true[sens == 0] == -1)) / np.sum(pos_0)
            fdr_1 = np.sum(pos_1 & (y_true[sens == 1] == -1)) / np.sum(pos_1)
            if fdr_0 == 0 or fdr_1 == 0:
                return 0
            return min(fdr_0 / fdr_1, fdr_1 / fdr_0)
        except:
            print("-->sens", sens)
            pos_0_1 = y_pred[sens[:,0] == 0] == 1
            pos_1_1 = y_pred[sens[:,0] == 1] == 1
            if np.sum(pos_0_1) == 0 or np.sum(pos_1_1) == 0:
                return 0
            fdr_0_1 = np.sum(pos_0_1 & (y_true[sens[:,0] == 0] == -1)) / np.sum(pos_0_1)
            fdr_1_1 = np.sum(pos_1_1 & (y_true[sens[:,0] == 1] == -1)) / np.sum(pos_1_1)
            if fdr_0_1 == 0 or fdr_1_1 == 0:
                return 0
            pos_0_2 = y_pred[sens[:,1] == 0] == 1
            pos_1_2 = y_pred[sens[:,1] == 1] == 1
            if np.sum(pos_0_2) == 0 or np.sum(pos_1_2) == 0:
                return 0
            fdr_0_2 = np.sum(pos_0_2 & (y_true[sens[:,1] == 0] == -1)) / np.sum(pos_0_2)
            fdr_1_2 = np.sum(pos_1_2 & (y_true[sens[:,1] == 1] == -1)) / np.sum(pos_1_2)
            if fdr_0_2 == 0 or fdr_1_2 == 0:
                return 0

            return (min(fdr_0_1 / fdr_1_1, fdr_1_1 / fdr_0_1) + min(fdr_0_2 / fdr_1_2, fdr_1_2 / fdr_0_2))/float(2)