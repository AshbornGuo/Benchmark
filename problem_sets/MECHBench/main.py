from src import sob
import numpy as np
import pandas as pd

# Import the platform
import platform


'''
Added the following block to identify if the current system 
'''

r'''
Once the optimization problem instance has been generate, 
the model is determined (mesh and fem data loaded) only when the variable array has been input.
'''

linux_system = not (platform.system() == 'Windows')

if linux_system:
    orss_main_path = "/home/ivanolar/Documents/OpenRadioss2/OpenRadioss_linux64/OpenRadioss/"
    
else:
    orss_main_path = r"C:/Users/guoji/Desktop/graduate project/codes/benchmarks/OpenRadioss_win64/OpenRadioss_win64/OpenRadioss"

runnerOptions = {"open_radioss_main_path":orss_main_path,
                 "write_vtk":True,
                 "np":1,
                 "nt":4,
                 "h_level":1,
                 "gmsh_verbosity":0,
}

def main():
    sim_id = 255 # Attribute to define the simulation id and connected results folder name
    vector = [3.42834656, 2.15259375, -1.49093763, 1.69472419, -0.29146718, 1.32146257, 4.98257611, -2.80885004, -3.23567450, -4.48471372]
    # vector = np.zeros((30,)).tolist()] # Vector where the objective function is evaluated, it has as many components as the second input argument in get_problem below
    
    #原代码行，测试暂时注释
    # f = sob.get_problem(3,10,runnerOptions,"load_uniformity",sequential_id_numbering=False)#shown case

    
    # f = sob.get_problem(3, 10, runnerOptions, "mass", sequential_id_numbering=False)#OK
    # f = sob.get_problem(1, 10, runnerOptions, "absorbed_energy", sequential_id_numbering=False)#OK
    # f = sob.get_problem(3, 10, runnerOptions, "intrusion", sequential_id_numbering=False)#OK
    # f = sob.get_problem(1, 10, runnerOptions, "specific_energy_absorbed", sequential_id_numbering=False)#OK
    # f = sob.get_problem(1, 10, runnerOptions, "mean_impact_force", sequential_id_numbering=False)# 2 OK  1，3 the same problem: force calculation
    # f = sob.get_problem(3, 10, runnerOptions, "peak-impact-force", sequential_id_numbering=False) #123not defined？
    # f = sob.get_problem(1, 10, runnerOptions, "usage_ratio", sequential_id_numbering=False) #nan
    # f = sob.get_problem(1, 10, runnerOptions, "load_uniformity", sequential_id_numbering=False)#2 nan  #13 the same problem: force calculation
    # f = sob.get_problem(1, 10, runnerOptions, "penalized_sea", sequential_id_numbering=False)   #1 2 invalid syntax syn# 3 not allow
    f = sob.get_problem(2, 10, runnerOptions, "penalized_mass", sequential_id_numbering=False)    # 2 OK  1 3 not allow




    obj_value = f(vector,sim_id)
    print(obj_value)
    

if __name__ == '__main__':
    main()


