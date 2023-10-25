from math import pi
from operator import add

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter  # PillowWriter


class Forces:
    """
    class that contains all the necessary forces for the
    """

    def __init__(self, radius, dt, rho=7.85e-6, x1=0.0, y1=0.0, x2=100.0, y2=100.0,
                 stiffness=1e2, exponent=2.1, dmax=0.01, damping_factor=None):  # c=0.1

        # mandatory parameters:
        self.r = radius
        self.dt = dt
        # Optional parameters:
        self.rho = rho

        # Box points (1-> left lower 2-> right upper):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        ec = 0.01*(x2-x1)
        self.x1c = x1 + ec
        self.y1c = y1 + ec
        self.x2c = x2 - ec
        self.y2c = y2 - ec

        # Contact properties
        self.stiffness = stiffness
        self.exponent = exponent
        self.dmax = dmax
        self.damping_factor = (0.1 * stiffness) / 100 if damping_factor is None else damping_factor
        # Initialized
        self.volume = self.get_volume()
        self.mass = self.get_mass()
        self.inertia = self.get_inertia()
        self.mass_properties = [self.mass, self.mass, self.inertia]

    def get_volume(self):
        return (4 / 3) * pi * self.r ** 3

    def get_mass(self):
        return self.volume * self.rho

    def get_inertia(self):
        return (2 / 5) * self.mass * self.r ** 2

    @staticmethod
    def clamp(x, lower_lim=0.0, upper_lim=1.0):
        """
        Clamp function limits the result to specific max and min.
        :param x: the value that will be checked.
        :param lower_lim: lower limit.
        :param upper_lim: upper limit.
        :return: return the value to be on the limits if the value is greater/lower than them.
        """
        if x < lower_lim:
            return lower_lim
        if x > upper_lim:
            return upper_lim
        return x

    def smooth_step(self, x, x0, x1, y0, y1):
        """
        Smooth step function. https://en.wikipedia.org/wiki/Smoothstep
        :param x: independent variable
        :param x0: x value in the start of step function.
        :param x1: x value in the end of step function.
        :param y0: y value in the start of step function.
        :param y1: y value in the end of step function.
        :return: the interpolated value through step function.
        """
        x = self.clamp((x - x0) / (x1 - x0))
        dy = y1 - y0
        return y0 + dy * (6 * (x ** 5) - 15 * (x ** 4) + 10 * (x ** 3))

    def contact_detection(self, x, y):
        """
        Contact detection function.
        # TODO: Currently not used.
        :param x: x position of the ball center.
        :param y: y position of the ball center.
        :return: the contact detection of the x, y and the direction of contact force
        0-> no contact, 1-> positive contact, -1-> negative contact
        """

        con_x = 0
        con_y = 0
        if (x + self.r) > self.x2:
            con_x = -1
            # pd_x = (x + self.r) - self.x2
        elif (x - self.r) < self.x1:
            con_x = 1
        if (y + self.r) > self.y2:
            con_y = -1
        elif (y - self.r) < self.y1:
            con_y = 1

        return [con_x, con_y]

    def penetration_depth(self, x, y):
        """
        Calculates the signed penetration depth of the contact.

        :param x: x position of the ball center.
        :param y: y position of the ball center.
        :return: the penetration depth for the contact in x and y direction.
        """
        pd_x = min(0.0, self.x2c - (x + self.r)) + max(0.0, self.x1c - (x - self.r))
        pd_y = min(0.0, self.y2c - (y + self.r)) + max(0.0, self.y1c - (y - self.r))

        return [pd_x, pd_y]

    def f_damp(self, pd, pd_cur):
        """
        Calculates the damping contact force for specific penetration depth and pen. depth velocity
        :param pd: contact penetration depth (future value).
        :param pd_cur: current contact penetration depth in time_index .
        :return: Damping contact force.
        """
        # return -self.smooth_step(pd, -self.dmax, self.damping_factor, 0.0, 0.0) * pd_dot
        # calculate penetration depth velocity:
        pd_dot = (pd - pd_cur) / self.dt
        # Damping should have the same direction with the pd_dot
        return self.damping_factor * pd_dot

    def f_stiff(self, pd):
        """
        Calculates the spring contact force.
        :param pd: penetration depth
        :return: spring contact force.
        """
        if pd == 0:
            return 0
        else:
            return (pd / abs(pd)) * self.stiffness * abs(pd) ** self.exponent
            # return (-dr / abs(dr)) * self.stiffness * abs(pd) ** self.exponent

    def f_contact(self, x, y, pd_current):  # dx, dy,
        """
        Total contact force.
        :param x: x position of the ball.
        :param y: y position of the ball
        # :param dx: dx/dt velocity of the ball.
        # :param dy: dy/dt velocity of the ball
        :param pd_current: the current penetration depth for time_index.
        :return: total contact force.
        """
        # TODO: check what should we do for the contact force due to the rotation

        f_stiff = map(self.f_stiff, self.penetration_depth(x, y))
        # f_stiff = [self.f_stiff(pd, dr) for pd, dr in zip(self.penetration_depth(x, y), [dx, dy])]

        f_damp = [self.f_damp(pd, pd_cur) for pd, pd_cur in zip(self.penetration_depth(x, y), pd_current)]

        results = list(map(add, f_stiff, f_damp))  # [ contact force on x, contact force on y   ]
        # or [x+y for x, y in zip(f_stiff, f_damp)]

        results.append(0.0)  # component of contact for the rotation generalized component

        return results

    def f_friction(self, mu, gravity=9.81):
        # TODO: mu is a function of velocity (?)
        # TODO: will be applied. skipped for now
        return mu * self.mass * gravity

    def f_ext(self, fx_in=0.0, fy_in=0.0, mzz_in=0.0):

        return [fx_in, fy_in, mzz_in]

    def f_air(self, dx, dy, cd=0.2, fluid='oil'):
        """
        :param dx: velocity x
        :param dy: velocity y
        :param cd: drag coefficient of the ball.
        :param fluid: type of the fluid.
        :return: Return the coefficient for drag force.
        """

        rho = 9.5e-7 if fluid == 'oil' else 1.293e-9  # oil and air density
        d_factor = 0.5 * cd * rho * pi * self.r ** 2
        sign_x = - dx / abs(dx) if abs(dx) > 0 else 0
        sign_y = - dy / abs(dy) if abs(dy) > 0 else 0
        return [sign_x * d_factor*dx**2, sign_y*d_factor*dy**2, 0.0]
        # return [-d_factor*dx**2, -d_factor*dy**2, 0.0]
        # return [0.0, 0.0, 0.0]


class Solver:
    """
    Class that contains all the necessary functionality for the solution of the ODE system.
    The solution is based on 4th order Runge Kutta method.
    """

    def __init__(self, parameters, t0=0.0):
        """Constructor"""

        self.parameters = parameters  # dictionary
        self.t0 = t0
        self.u0 = np.array([parameters['initial_x_pos'],
                            parameters['initial_y_pos'],
                            0.0,
                            parameters['initial_x_vel'],
                            parameters['initial_y_vel'],
                            0.0])
        self.t_end = parameters['simulation_end_time']
        self.dt = parameters['time_step']
        self.radius = parameters['radius']
        self.forces = Forces(self.radius, self.dt)

        self.func = []
        # initialize penetration depth list of list (x,y). useful for calculating penetration depth velocity
        self.pd = [self.forces.penetration_depth(self.u0[0], self.u0[1])]  # [[0.0, 0.0]]
        # index. calculates the number of time steps
        self.time_index = 0

    def solve(self):
        """
        Solve the ODE with 4th order Runge Kutta method.
        :return: results list of list (size mx7 where m is defined as the time index and 7 is for t,u1:u6).
        """
        self.func = [self.f1, self.f2, self.f3, self.f4, self.f5, self.f6]
        return self.runge_kutta_4(self.func, self.u0, self.t0, self.t_end, self.dt)

    # Example usage
    def f1(self, t, u1, u2, u3, u4, u5, u6):
        """
        Function for u1 (right part on the equation du/dt = f(u) where u = [u1, u2, u3, u4, u5, u6]
        :param t: time
        :param u1: u1 = x
        :param u2: u2 = y
        :param u3: u3 = theta_z
        :param u4: u4 = dx/dt
        :param u5: u5 = dy/dt
        :param u6: u6 = dtheta_z/dt
        :return: the value of the f(u)[0]
        """
        # Define the first differential equation
        return u4

    def f2(self, t, u1, u2, u3, u4, u5, u6):
        """
        Function for u2 (right part on the equation du/dt = f(u) where u = [u1, u2, u3, u4, u5, u6]
        :param t: time
        :param u1: u1 = x
        :param u2: u2 = y
        :param u3: u3 = theta_z
        :param u4: u4 = dx/dt
        :param u5: u5 = dy/dt
        :param u6: u6 = dtheta_z/dt
        :return: the value of the f(u)[1]
        """
        # Define the second differential equation
        return u5

    def f3(self, t, u1, u2, u3, u4, u5, u6):
        """
        Function for u3 (right part on the equation du/dt = f(u) where u = [u1, u2, u3, u4, u5, u6]
        :param t: time
        :param u1: u1 = x
        :param u2: u2 = y
        :param u3: u3 = theta_z
        :param u4: u4 = dx/dt
        :param u5: u5 = dy/dt
        :param u6: u6 = dtheta_z/dt
        :return: the value of the f(u)[2]
        """
        # Define the third differential equation
        return u6

    def f4(self, t, u1, u2, u3, u4, u5, u6):
        """
        Function for u4 (right part on the equation du/dt = f(u) where u = [u1, u2, u3, u4, u5, u6]
        :param t: time
        :param u1: u1 = x
        :param u2: u2 = y
        :param u3: u3 = theta_z
        :param u4: u4 = dx/dt
        :param u5: u5 = dy/dt
        :param u6: u6 = dtheta_z/dt
        :return: the value of the f(u)[3]
        """
        return self.calculate_forces(t, u1, u2, u3, u4, u5, u6, 0)

    def f5(self, t, u1, u2, u3, u4, u5, u6):
        """
        Function for u5 (right part on the equation du/dt = f(u) where u = [u1, u2, u3, u4, u5, u6]
        :param t: time
        :param u1: u1 = x
        :param u2: u2 = y
        :param u3: u3 = theta_z
        :param u4: u4 = dx/dt
        :param u5: u5 = dy/dt
        :param u6: u6 = dtheta_z/dt
        :return: the value of the f(u)[4]
        """
        return self.calculate_forces(t, u1, u2, u3, u4, u5, u6, 1)

    def f6(self, t, u1, u2, u3, u4, u5, u6):
        """
        Function for u6 (right part on the equation du/dt = f(u) where u = [u1, u2, u3, u4, u5, u6]
        :param t: time
        :param u1: u1 = x
        :param u2: u2 = y
        :param u3: u3 = theta_z
        :param u4: u4 = dx/dt
        :param u5: u5 = dy/dt
        :param u6: u6 = dtheta_z/dt
        :return: the value of the f(u)[5]
        """
        return self.calculate_forces(t, u1, u2, u3, u4, u5, u6, 2)

    def calculate_forces(self, t, u1, u2, u3, u4, u5, u6, force_j):
        """
        Common function for the u3, u4, u5 coordinates
        :param t: time
        :param u1: u1 = x
        :param u2: u2 = y
        :param u3: u3 = theta_z
        :param u4: u4 = dx/dt
        :param u5: u5 = dy/dt
        :param u6: u6 = dtheta_z/dt
        :param force_j: choose between 0, 1, 2 for calculating force for x, y, theta_z accordingly
        :return: Calculated force for x,y,theta_z depending on force_j
        """
        x = u1
        y = u2
        dx = u4
        dy = u5
        # self.pd.append(self.forces.penetration_depth(x, y))
        # pd_dot = [(self.pd[self.time_index][i] - self.pd[self.time_index - 1][i]) / self.dt for i in [0, 1]]
        pd_current = self.pd[self.time_index]
        result = (1 / self.forces.mass_properties[force_j]) * (
            self.forces.f_contact(x, y, pd_current)[force_j] +  # dx, dy,
            self.forces.f_air(dx, dy)[force_j]
        )
        # TODO: here all the other forces will be added

        return result

    def runge_kutta_4(self, func, u0, t0, t_end, h):
        """
        Solution of the 4th order Runge Kutta method
        :param func: the right part of the equation du/dt = f(u)
        :param u0: initial conditions for u
        :param t0: initial time.
        :param t_end: end time of the simulation.
        :param h: the step size of the simulation.
        :return: Calculated results for every time step. ut = [ ]
        """
        t = t0
        u = u0
        results = [(t, *u)]
        self.time_index = 0
        while t < t_end:
            k1 = [h * f(t, *u) for f in func]  # size 3 should be 6
            k2 = [h * f(t + h / 2, *(u[i] + k1[i] / 2 for i in range(len(u)))) for f in func]
            k3 = [h * f(t + h / 2, *(u[i] + k2[i] / 2 for i in range(len(u)))) for f in func]
            k4 = [h * f(t + h, *(u[i] + k3[i] for i in range(len(u)))) for f in func]

            u = [u[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(len(u))]
            t += h

            self.pd.append(self.forces.penetration_depth(u[0], u[1]))

            self.time_index += 1
            results.append((t, *u))

        return results


class Plot:
    """Class that contains functionality about plotting the results and saving animations"""

    def __init__(self, results, parameters):
        """Constructor"""
        self.results = results
        self.radius = parameters['radius']

    def plot_results(self):
        """ PLot both static and animated results"""
        self.plot_static()
        self.plot_animation()

    def plot_static(self):
        """Plot static figures"""
        t_values = [result[0] for result in self.results]
        u1_values = [result[1] for result in self.results]
        u2_values = [result[2] for result in self.results]
        u3_values = [result[3] for result in self.results]
        u4_values = [result[4] for result in self.results]
        u5_values = [result[5] for result in self.results]
        u6_values = [result[6] for result in self.results]

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(t_values, u1_values, label='u1', linestyle='dotted')
        plt.plot(t_values, u2_values, label='u2', linestyle='dotted')
        plt.plot(t_values, u3_values, label='u3', linestyle='dotted')
        plt.xlabel('Time')
        plt.ylabel('u1, u2, u3')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t_values, u4_values, label='u4')
        plt.plot(t_values, u5_values, label='u5')
        plt.plot(t_values, u6_values, label='u6')
        plt.xlabel('Time')
        plt.ylabel('u4, u5, u6')
        plt.legend()

        plt.tight_layout()

        plt.figure(figsize=(8, 6))

        plt.plot(u1_values, u2_values)
        plt.xlabel('u1')
        plt.ylabel('u2')
        plt.title('u1 vs u2')

        plt.show()

    def plot_animation(self):
        t_values = [result[0] for result in self.results]
        u1_values = [result[1] for result in self.results]
        u2_values = [result[2] for result in self.results]

        fig, ax = plt.subplots()  # figsize=(8, 6)

        print(len(u1_values))

        # TODO: interpolation didn't worked this way. We want unsorted interpolation.
        # Interpolate the data to create smoother curves
        # u1_interp = np.linspace(min(u1_values), max(u1_values), 2000)
        # u2_interp = np.interp(u1_interp, u1_values, u2_values)
        # print(len(u2_interp))

        [rec_x, rec_y, rec_width, rec_height] = [0, 0, 100, 100]
        rectangle = patches.Rectangle((rec_x, rec_y), rec_width, rec_height,
                                      linewidth=1, edgecolor='r', facecolor='none')

        def update(frame):
            """Function that run for every frame and creates it."""
            ax.clear()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('DynoBall path')

            # ax.plot(u1_interp[:frame + 1], u2_interp[:frame + 1])
            ax.plot(u1_values[:frame + 1], u2_values[:frame + 1])

            ax.add_patch(rectangle)

            circle = plt.Circle((u1_values[frame], u2_values[frame]), self.radius,
                                fill=False, color='blue', linestyle='dashed')
            ax.add_patch(circle)
            fig.tight_layout()

        len_frames = int(len(t_values) - 1)
        ani = FuncAnimation(fig, update, frames=len_frames, repeat=False, interval=1)

        # To save the animation using Pillow as a gif
        # writer = PillowWriter(fps=1000,
        #                       metadata=dict(artist='mylopouc'),
        #                       bitrate=1800)

        # Set up an FFMpegWriter
        writer = FFMpegWriter(fps=100, metadata=dict(artist='mylopouc'))
        plt.show()

        # ani.save('DynoBall_results.gif', writer='pillow', fps=150, metadata=dict(artist='mylopouc'))
        ani.save('DynoBall_results.mp4', writer=writer)


if __name__ == "__main__":
    pass
    # # print("Hello World")
    # width = 100.0
    # epsilon = 1.0
    # const_product = 1.0
    # lower = 0.0
    # current_x = 99.1
    #
    # """Test for upper limit."""
    # myForce = Forces(radius=10)
    # print(myForce.smooth_step(99.5, width - epsilon, width, 5.0, 10.0))
    #
    # """Test for lower limit."""
    #
    # print(myForce.smooth_step(0.5, 0.0, epsilon, 10.0, 5.0))
    #
    # print(myForce.f_contact(current_x, lower, width, epsilon, const_product))
    #
    # """ Plot the results for range of x"""
    #
    # # Generate x values
    # x_values = np.linspace(-1, 3, 1000)
    #
    # # Calculate y values
    # y_values = [myForce.f_contact(x, lower, width, epsilon, const_product) for x in x_values]
    #
    # # Plot the function
    # plt.plot(x_values, y_values)
    # plt.title('f_contact(x)')
    # plt.xlabel('x')
    # plt.ylabel('f_contact(x)')
    # plt.show()
