import numpy as np
import time
import torch

from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination

from botorch.exceptions import BadInitialCandidatesWarning
import warnings
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


def generate_initial_x(problem, n_init):
    unit_bounds = torch.stack([
        torch.zeros(problem.n_var, **tkwargs),
        torch.ones(problem.n_var, **tkwargs)
    ])
    x_unit = draw_sobol_samples(bounds=unit_bounds, n=n_init, q=1).squeeze(1)
    return unnormalize(x_unit, bounds=problem.bounds)


def build_model(train_x, train_obj_noisy, train_con_noisy, problem):
    train_x_gp = normalize(train_x, problem.bounds)
    train_y = torch.cat([train_obj_noisy, train_con_noisy], dim=-1)

    models = []
    for i in range(train_y.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x_gp,
                train_y[..., i:i+1],
                outcome_transform=Standardize(m=1),
            )
        )

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll, train_x_gp


def create_constraint_callables(problem):
    def create_idxr(i):
        def idxr(Z):
            return Z[..., i]
        return idxr

    return [
        create_idxr(i)
        for i in range(problem.n_obj, problem.n_obj + problem.n_constr)
    ]


def compute_hv(train_obj, train_con, ref_point):
    hv = Hypervolume(ref_point=-ref_point)

    is_feas = (train_con <= 0).all(dim=-1)
    feas_obj = train_obj[is_feas]

    if feas_obj.shape[0] == 0:
        return 0.0

    pareto_mask = is_non_dominated(feas_obj)
    pareto_y = feas_obj[pareto_mask]
    volume = hv.compute(pareto_y)
    return float(volume)


def optimize_egbo(
    problem,
    ref_point,
    initial_x,
    n_batch,
    batch_size,
    random_state=0,
    noise=0.0,
    verbose=True,
):
    print("Optimizing with EGBO (qNEHVI + U-NSGA-III)")
    torch.manual_seed(random_state)

    t0 = time.time()
    hvs = []

    train_x = initial_x
    train_obj, train_con = problem.evaluate(train_x)

    train_obj_noisy = train_obj + noise * torch.randn_like(train_obj)
    train_con_noisy = train_con + noise * torch.randn_like(train_con)

    model, mll, train_x_gp = build_model(train_x, train_obj_noisy, train_con_noisy, problem)

    unit_bounds = torch.stack([
        torch.zeros(problem.n_var, **tkwargs),
        torch.ones(problem.n_var, **tkwargs)
    ])

    constraint_callables = create_constraint_callables(problem)

    for iteration in range(1, n_batch + 1):
        t_iter_0 = time.time()

        fit_gpytorch_mll(mll)

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=-ref_point,
            X_baseline=train_x_gp,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
            objective=IdentityMCMultiOutputObjective(
                outcomes=np.arange(problem.n_obj).tolist()
            ),
            constraints=constraint_callables,
            prune_baseline=True,
            cache_pending=True,
        )

        qnehvi_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=unit_bounds,
            q=batch_size,
            num_restarts=1,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
        )

        pareto_mask = is_non_dominated(train_obj)
        pareto_x = train_x_gp[pareto_mask]
        pareto_y = -train_obj[pareto_mask]
        pareto_con = train_con[pareto_mask]

        if pareto_x.shape[0] < 2:
            nsga_x = draw_sobol_samples(bounds=unit_bounds, n=256, q=1).squeeze(1)
        else:
            algorithm = UNSGA3(
                pop_size=256,
                ref_dirs=get_reference_directions(
                    "energy", problem.n_obj, batch_size, seed=random_state
                ),
                sampling=pareto_x.detach().cpu().numpy(),
            )

            pymooproblem = PymooProblem(
                n_var=problem.n_var,
                n_obj=problem.n_obj,
                n_constr=problem.n_constr,
                xl=np.zeros(problem.n_var),
                xu=np.ones(problem.n_var),
            )

            algorithm.setup(pymooproblem, termination=NoTermination())

            pop = algorithm.ask()
            pop.set("F", pareto_y.detach().cpu().numpy())
            pop.set("G", pareto_con.detach().cpu().numpy())
            algorithm.tell(infills=pop)

            newpop = algorithm.ask()
            nsga_x = torch.tensor(newpop.get("X"), **tkwargs)

        candidates = torch.cat([qnehvi_x, nsga_x], dim=0)
        acq_values = []

        for i in range(candidates.shape[0]):
            with torch.no_grad():
                v = acq_func(candidates[i].unsqueeze(0))
                acq_values.append(v.item())

        sorted_idx = np.argsort(acq_values)
        best_x_unit = candidates[sorted_idx[-batch_size:]]
        new_x = unnormalize(best_x_unit, bounds=problem.bounds)

        new_obj, new_con = problem.evaluate(new_x)
        new_obj_noisy = new_obj + noise * torch.randn_like(new_obj)
        new_con_noisy = new_con + noise * torch.randn_like(new_con)

        train_x = torch.cat([train_x, new_x], dim=0)
        train_obj = torch.cat([train_obj, new_obj], dim=0)
        train_con = torch.cat([train_con, new_con], dim=0)
        train_obj_noisy = torch.cat([train_obj_noisy, new_obj_noisy], dim=0)
        train_con_noisy = torch.cat([train_con_noisy, new_con_noisy], dim=0)

        volume = compute_hv(train_obj, train_con, ref_point)
        hvs.append(volume)

        model, mll, train_x_gp = build_model(train_x, train_obj_noisy, train_con_noisy, problem)

        t_iter_1 = time.time()
        if verbose:
            n_feas = int(((train_con <= 0).all(dim=-1)).sum().item())
            print(
                f"Batch {iteration:>2}/{n_batch} | "
                f"HV = {hvs[-1]:.4f} | "
                f"feasible = {n_feas} | "
                f"time = {t_iter_1 - t_iter_0:.2f}s"
            )

    total_time = time.time() - t0
    print(f"Total time: {total_time:.2f}s")

    train_data = torch.hstack([train_x, train_obj, train_con]).detach().cpu().numpy()
    return hvs, train_data, total_time


#############################################

class SimpleProblem(torch.nn.Module):
    n_var = 2
    n_obj = 2
    n_constr = 1

    bounds = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0]
    ], **tkwargs)

    # IMPORTANT:
    # original objectives are minimized and roughly lie in [0, 2]
    # so choose a WORSE reference point in original minimization space
    ref_point = torch.tensor([2.5, 2.5], **tkwargs)

    @staticmethod
    def evaluate(X):
        x1 = X[:, 0]
        x2 = X[:, 1]

        # original minimization objectives
        f1 = x1**2 + x2**2
        f2 = (x1 - 1.0)**2 + (x2 - 1.0)**2

        # feasible if <= 0
        c1 = x1 + x2 - 1.5

        # convert minimization to maximization
        obj = torch.stack([-f1, -f2], dim=-1)
        con = c1.unsqueeze(-1)

        return obj, con
    

#####################################

problem = SimpleProblem

n_init = 6   # 2 * (d + 1) for d=2
initial_x = generate_initial_x(problem, n_init=n_init)

hvs, train_data, total_time = optimize_egbo(
    problem=problem,
    ref_point=problem.ref_point,
    initial_x=initial_x,
    n_batch=10,
    batch_size=4,
    random_state=0,
    noise=0.0,
    verbose=True,
)

print("Final HV list:")
print(hvs)

print("Train data shape:")
print(train_data.shape)