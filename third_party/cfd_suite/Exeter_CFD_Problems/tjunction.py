from data.SnappyHexOptimise import BasicTRun
from interfaces import ControlPolygonInterface
from base_class import Problem
import data.support as support
import numpy as np

class TJunction(Problem, ControlPolygonInterface):

    def __init__(self, **settings):
        self.source_case = settings.get('source_case', 'data/TJunction/case_T')
        self.case_path = settings.get('case_path', 'data/TJunction/case_multi/')
        self.domain_files = settings.get('boundary_files', \
                                    ['data/TJunction/boundary_1.csv',\
                                    'data/TJunction/boundary_2.csv',\
                                    'data/TJunction/boundary_3.csv'])
        self.fixed_points_files = settings.get('fixed_points_files', \
                                    ['data/TJunction/fixed_1.csv',\
                                    'data/TJunction/fixed_2.csv',\
                                    'data/TJunction/fixed_3.csv'])
        self.n_control = settings.get('n_control', [2, 2, 2])
        self.niter = settings.get('niter', 5)
        self.thickness = settings.get('thickness', np.array([0, 0, 0.1]))
        self.stl_dir = settings.get('stl_dir', 'constant/triSurface/')
        self.stl_file_name = settings.get('stl_file_name', 'ribbon.stl')
        self.setup()

    def setup(self, verbose=False):
        pts = [np.loadtxt(filename, delimiter=',') for filename in self.domain_files]
        fixed_points = [list(np.loadtxt(filename, delimiter=',').astype(int).tolist())\
                            for filename in self.fixed_points_files]
        '''
        fixed_points = []
        #[[list(j) for j in f]] if len(f.shape)<2 else list(f) for f in fixed_points]
        for f in fixed_points:
            if len(f.shape)<2:
                fixed_points.append(list())
        '''
        #import pdb; pdb.set_trace()
        #fixed_points = [[[0], [4]], [[0], [3]], [[0], [4]]]
        cpolys = []
        for i in range(len(pts)):
            cpolys.append(support.ControlPolygon2D(pts[i], fixed_points[i], self.n_control[i]))
        ControlPolygonInterface.__init__(self, cpolys)
        problem = BasicTRun(case_path=self.case_path)
        problem.prepare_case(self.source_case, verbose)
        self.problem = problem

    def info(self):
        pass

    def get_configurable_settings(self):
        pass

    def run(self, shape):
        curv_data = support.subdc_to_stl_mult(shape, self.niter,\
                    thickness=self.thickness,\
                    file_directory=self.case_path+self.stl_dir, \
                    file_name=self.stl_file_name,\
                    draw=False)
        p1, p2 = self.problem.cost_function(sense="multi")
        return np.abs(p1), np.abs(p2)

    def evaluate(self, decision_vector):
        shape = self.convert_decision_to_shape(decision_vector)
        return self.run(shape)

if __name__=='__main__':
    import numpy as np
    seed = 12345
    np.random.seed(seed)
    prob = TJunction()
    lb, ub = prob.get_decision_boundary()
    x = np.random.random((1000, lb.shape[0])) * (ub - lb) + lb
    rand_x = []
    for i in range(x.shape[0]):
        if prob.constraint(x[i]):
            rand_x.append(x[i])
    #res = prob.evaluate(rand_x[0])
    #print(res)
