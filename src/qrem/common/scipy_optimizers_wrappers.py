"""
Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""
from typing import Callable, List, Tuple, Optional

import numpy as np
from scipy import optimize as scoptimize
from scipy.optimize import LinearConstraint


from qrem.common.printer import qprint

def wrapper_scipy_optimize(
        target_function: Callable,
        optimizer_name: str,
        initial_parameters: Optional[List[float]]=None,
        basinhopping=False,
        basinhopping_kwargs={},
        differential_evolution_kwargs={},
        additional_arguments: Optional[Tuple] = (),
        bounds_list: Optional[List[Tuple[float,float]]] = None,
        options: Optional[dict] = None,
        differential_evolution_constraints = ()
):


    if optimizer_name.upper() not in ['DIFFERENTIAL_EVOLUTION'] and initial_parameters is None:
        raise ValueError(f"Optimizer '{optimizer_name}' requires initial parameters guess!")


    if optimizer_name.upper() in ['DIFFERENTIAL_EVOLUTION'] and basinhopping:
        raise ValueError("Chosen optimizer 'differential evolution' does not support basinhopping,"
                         " but it was set to True.")


    number_of_parameters = len(initial_parameters)
    if bounds_list is not None:
        if optimizer_name in ['COBYLA', 'SLSQP']:
            constraints_list = []
            for lower, upper in bounds_list:
                def _ineq_fuction_now(x):
                    return (lower - x) * (x - upper)

                constraints_list.append({'type': 'ineq',
                                         'fun': _ineq_fuction_now})
            bounds_list = None

        elif optimizer_name == 'trust-constr':
            constraints_list = LinearConstraint(np.eye(number_of_parameters),
                                                lb=[tup[0] for tup in bounds_list],
                                                ub=[tup[1] for tup in bounds_list])

        else:
            bounds_list = bounds_list
            constraints_list = ()
    else:
        constraints_list = ()

    if basinhopping:

        res = scoptimize.basinhopping(func=target_function,
                                      x0=initial_parameters,
                                      **basinhopping_kwargs,
                                      minimizer_kwargs={'method': optimizer_name,
                                                        'options': options,
                                                        'args': additional_arguments,
                                                        'constraints':constraints_list,
                                                        'bounds':bounds_list}
                                      # minimizer_kwargs=dict,
                                      )

    else:
        if optimizer_name.upper() == 'DIFFERENTIAL_EVOLUTION':
            # print('printing hej')
            res = scoptimize.differential_evolution(func=target_function,
                                                    bounds=bounds_list,
                                                    args=additional_arguments,
                                                    constraints=differential_evolution_constraints,
                                                    **differential_evolution_kwargs
                                                    )
            # print('printing hej2')

        else:

            res = scoptimize.minimize(fun=target_function,
                                      x0=initial_parameters,
                                      args=additional_arguments,
                                      method="trust-constr",
                                      bounds=bounds_list,
                                      constraints=constraints_list,
                                      options=options
                                      )

    return res


def fun_test(xs):
    return 2 * xs[0] + 3 * xs[1]


if __name__ == '__main__':
    params_starting = [1, 1]

    number_of_parameters = len(params_starting)

    bounds_list = [(0, 1) for _ in range(number_of_parameters)]

    optimizer_name = 'Nelder-Mead'
    options = {'maxiter': 10 ** 7}
    res = wrapper_scipy_optimize(target_function=fun_test,
                                 initial_parameters=params_starting,
                                 bounds_list=bounds_list,
                                 optimizer_name=optimizer_name,
                                 options=options)

    optimal_parameters = res.x
    optimal_value = res.fun

    qprint('Optimal parameters:', optimal_parameters)
    qprint('Optimal value:', optimal_value)

    res = wrapper_scipy_optimize(target_function=fun_test,
                                 initial_parameters=optimal_parameters,
                                 bounds_list=bounds_list,
                                 optimizer_name=optimizer_name)
    print()
    optimal_parameters = res.x
    optimal_value = res.fun

    qprint('Optimal parameters:', optimal_parameters)
    qprint('Optimal value:', optimal_value)
