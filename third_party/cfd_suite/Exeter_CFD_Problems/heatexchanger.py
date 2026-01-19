try:
    from data.SnappyHexOptimise import BasicHeatExchangerRun
except:
    from .data.SnappyHexOptimise import BasicHeatExchangerRun
try:
    from interfaces import PipeInterface
except:
    from .interfaces import PipeInterface
try:
    from base_class import Problem
except:
    from .base_class import Problem
try:
    from data import support #import data.support as support
except:
    from .data import support #as support
import numpy as np

class HeatExchanger(Problem, PipeInterface):

    def __init__(self, settings):
        self.source_case = settings.get('source_case', 'data/HeatExchanger/heat_exchange')
        self.case_path = settings.get('case_path', 'data/HeatExchanger/case_multi/')
        self.stl_dir = settings.get('stl_dir', 'constant/triSurface/')
        self.stl_file_name = settings.get('stl_file_name', 'ribbon.stl')
        self.n_rows = 3
        self.nlb = settings.get('min_num_pipes_per_row', 1)
        self.nub = settings.get('max_num_pipes_per_row', 3)
        self.n_coeffs_radii = settings.get('num_coeff_rad_each_row', [2]*self.n_rows)
        assert len(self.n_coeffs_radii) == self.n_rows, 'Invalid number of coefficients for radii.'
        self.n_coeffs_num = settings.get('num_coeff_num_pipe', 4)
        self.n_betas = settings.get('num_betas_each_row', [2]*self.n_rows)
        assert len(self.n_betas) == self.n_rows, 'Invalid number of betas.'
        self.setup()

    def setup(self, verbose=False):
        self.D = 0.2
        self.vert_origin = 0
        self.vert_positions = np.array([-self.D, 0, self.D])
        self.xlb, self.xub = -self.D, 3.25*self.D
        self.rlb, self.rub = 0.005, 0.5*self.D
        PipeInterface.__init__(self, self.vert_origin, self.vert_positions, \
                                self.xlb, self.xub, self.rlb, self.rub, \
                                self.nlb, self.nub, self.n_coeffs_radii,\
                                self.n_coeffs_num, self.n_betas)
        problem = BasicHeatExchangerRun(case_path=self.case_path)
        problem.prepare_case(self.source_case, verbose)
        self.problem = problem

    def info(self):
        raise NotImplementedError

    def get_configurable_settings(self):
        raise NotImplementedError

    def run(self, shape, verbose=False):
        xp, yp, rp = shape
        support.circle_to_stl(rp, xp, yp, \
            file_directory=self.case_path+self.stl_dir, file_name=self.stl_file_name, draw=False)
        t, p = self.problem.cost_function(sense="multi", verbose=verbose)
        return t, p

    def evaluate(self, decision_vector, verbose=False):
        if not self.constraint(decision_vector):
            raise ValueError('Constraint violated. Please supply a feasible decision vector.')
        shape = self.convert_decision_to_shape(decision_vector)
        try:
            return self.run(shape, verbose)
        except Exception as e:
            print('Solution evluation failed.')
            print(e)

if __name__=='__main__':
    import numpy as np
    seed = 1435
    np.random.seed(seed)
    prob = HeatExchanger({})
    lb, ub = prob.get_decision_boundary()
    x = np.random.random((1000, lb.shape[0])) * (ub - lb) + lb
    rand_x = []
    for i in range(x.shape[0]):
        if prob.constraint(x[i]):
            rand_x.append(x[i])
    import time
    st = time.time()
    res = prob.evaluate(rand_x[0], verbose=True)
    print(res, (time.time() - st)/60)
