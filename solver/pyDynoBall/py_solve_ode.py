from math import exp
import matplotlib.pyplot as plt
import numpy as np


def clamp(x, lower_lim=0.0, upper_lim=1.0):
    if x < lower_lim:
        return lower_lim
    if x > upper_lim:
        return upper_lim
    return x


def smooth_step(x, x0, x1, y0, y1):
    x = clamp((x - x0) / (x1 - x0))
    dy = y1 - y0
    return y0 + dy * (6 * (x ** 5) - 15 * (x ** 4) + 10 * (x ** 3))


def f_contact(x, limit_down, limit_up, eps, c):

    # TODO: correction on the contact model. Consistent with Impact contact model of MotionSolve.
    # TODO: Make these reusable, different contacts for each of the boundaries (x_down, x_up, y_down, y_up)

    return (smooth_step(x, limit_down, limit_down + eps, 1, 0) * c * exp(x - (limit_down + eps)) +
            smooth_step(x, limit_up - eps, limit_up, 0, 1) * c * exp(limit_up - eps - x))


if __name__ == "__main__":
    # print("Hello World")
    width = 100.0
    epsilon = 1.0
    const_product = 1.0
    lower = 0.0
    current_x = 99.1

    """Test for upper limit."""

    print(smooth_step(99.5, width - epsilon, width, 5.0, 10.0))

    """Test for lower limit."""

    print(smooth_step(0.5, 0.0, epsilon, 10.0, 5.0))

    print(f_contact(current_x, lower, width, epsilon, const_product))

    """ Plot the results for range of x"""

    # Generate x values
    x_values = np.linspace(-1, 3, 1000)

    # Calculate y values
    y_values = [f_contact(x, lower, width, epsilon, const_product) for x in x_values]

    # Plot the function
    plt.plot(x_values, y_values)
    plt.title('f_contact(x)')
    plt.xlabel('x')
    plt.ylabel('f_contact(x)')
    plt.show()
