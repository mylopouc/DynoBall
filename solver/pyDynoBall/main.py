# This is a sample Python script.
from py_solve_ode import Solver, Plot


def run_game(dict_parameters):

    solver = Solver(dict_parameters)

    plot = Plot(solver.solve(), dict_parameters)

    plot.plot_results()


if __name__ == '__main__':
    parameters = {
        'radius': 10.0,
        'initial_x_pos': 50.0,
        'initial_y_pos': 80.0,
        'initial_x_vel': 50.0,
        'initial_y_vel': 150.0,
        'simulation_end_time': 10.0,
        'time_step': 0.02,
    }
    run_game(parameters)
