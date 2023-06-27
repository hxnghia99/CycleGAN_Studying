import numpy as np
import cvxpy as cp

x = cp.Variable((3,1))

P = np.array([[13, 12, -2], [12, 17, 6],[-2, 6, 12]], dtype=np.float32)
q = np.array([[-22], [-14.5], [13]], dtype=np.float32)
r = 1.0


objective = cp.Minimize(0.5*cp.quad_form(x,P)+np.transpose(q)@x+r)
constraints = [cp.abs(x)<=1]

prob = cp.Problem(objective, constraints)

prob.solve(verbose=True)

print(prob.variables)
