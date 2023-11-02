# DynoBall

"""A game for real life ball movement."""

Game instructions:

Give the initial conditions of the ball and check the ball motion inside a 
box.

This application will consist:

* Solver in c++ (currently only Python is available)
* Python interface 
* Connection between them (call C++ with Python)


Working environment:

* Git/GitHub
* Visual Studio (C++)
* PyCharm (Python)
  

## Solver
* C++ / Python language
* Formulation with Lagrange equations.
* Use Runge-Kutta 4th order explicit integration
* Impact contact interaction between ball and the box
* Drag force and friction force included
* Export an Animation
  
## Interface

* Use tkinter and PIL (Work in progress)
* Ask for the initial conditions and export the animation to the user

## Connection

* Calling c++ DLL from Python (TODO)

Result Animation:
![Animation results](https://github.com/mylopouc/DynoBall/assets/143400541/9951e28b-cf91-48d3-b7b8-fb55a8fe0ddc)







