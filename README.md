# reactor-design

Reactors are vessels that convert chemical reactants into chemical products. Predictive models are created from the continuity equation and the models are parameterized using 
experimental data or thermodynamic databases. 

cstr_temperature_effects is a constant stirred tank reactor. The model is comprised of a dynamic mass, energy and coolant balance. The model is parameterized using 
experimental data and the markov chain monte carlo - delayed rejection adaptive metropolis technique . The experimental data was fit to basis
spline functions so that the data could be differentiated and directly fit to the dynamic model instead of solving the differential equations at each iteration making the solver
much faster.

