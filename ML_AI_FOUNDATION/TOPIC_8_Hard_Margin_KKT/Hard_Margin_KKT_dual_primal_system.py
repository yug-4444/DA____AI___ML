"""
Future Work:
- Automatic support-vector detection.
- Full QP-based SVM training.
- Soft-margin SVM (C parameter).
- SMO algorithm.
- Kernel SVM (Polynomial/RBF).
- Nonlinear classification.
- Decision-boundary visualization.
"""

import numpy as np
from sklearn.svm import SVC
"""
    Special-Case Hard-Margin Linear SVM Solver

    Assumption:
        Every training point supplied to the model is a Support Vector.
        Under this assumption all margin constraints are active:

            y_i (w^T x_i + b) = 1

    Theory:
        Using the dual representation

            w = Σ alpha_i y_i x_i

        and defining the Signed Gram Matrix

            G_ij = y_i y_j (x_i^T x_j),

        the KKT conditions become

            G alpha + yb = 1
            y^Talpha = 0

        which can be written as the saddle-point linear system

            [ G   y ] [alpha] = [1]
            [y^T  0 ] [b]   [0]

        Solving this system directly yields the optimal dual
        variables alpha and intercept b, from which w is recovered.

    Notes:
        - This is NOT a general SVM solver.
        - It assumes the support vectors are already known.
        - Many real datasets contain non-support vectors (alpha = 0),
          which this formulation does not discover.
        - Intended as an educational implementation illustrating
          KKT conditions, duality, Gram matrices, and primal-dual
          reconstruction for linear SVMs.
    """
class SVCaseLinearSVM:#dataset input should all be of the support vectors only
    #"Find the separator assuming all points are support vectors."
    def __init__(self,x,y):
        self.x=np.array(x)#mxn
        if self.x.ndim==1:
            self.x=self.x.reshape(-1,1)#mx1
        self.y=np.array(y).reshape(-1,1)#mx1
        self.m,self.n=self.x.shape
        self.w=None
        self.b=None
        self.x_signed=self.y*self.x
        self.G=self.x_signed@self.x_signed.T
        self.alpha=None
        self.P_obj=None
        self.D_obj=None

    def solve(self):
        A=np.block([[self.G,self.y],[self.y.T,np.zeros((1,1))]])
        r=np.vstack([np.ones((self.m,1)),np.zeros((1,1))])
        sol=np.linalg.pinv(A)@r
        self.alpha=sol[:self.m]
        self.b=sol[self.m,0]
        self.w=self.x_signed.T@self.alpha

    def verifiy_duality(self):
        if self.alpha is None:
            raise ValueError("Run solve() first.")
        self.P_obj=1/2*(self.w.T@self.w).item()
        self.D_obj=np.sum(self.alpha)-1/2*(self.alpha.T@self.G@self.alpha).item()
        print(f"primal objective fun : {float(self.P_obj)}")
        print(f"dual objective fun : {float(self.D_obj)}")
        if self.D_obj<self.P_obj or np.allclose(self.P_obj,self.D_obj,atol=1e-6):
            print("also be obseverd that D*<=P* always as D* is tightest possible lower bound for P*\n")
        else:
            print("error occurend in formulation as voilataion of D*<*P*\n")
            return
        if np.allclose(self.P_obj,self.D_obj,atol=1e-6):
            print("strong duality P*=D*\n")
            strong_dual_w_norm=np.sqrt(np.sum(self.alpha))
            if np.allclose(strong_dual_w_norm,np.linalg.norm(self.w),atol=1e-6):
                print("->->-> Under strong duality then observed that norm w square =sum alpha verified\n")
            else:
                print("error occured in calculation of alpha\n")
                return
        else:
             print("weak duality D*<P*\n")
        b_veri=np.mean(self.y-self.x@self.w)
        if np.allclose(b_veri,self.b,atol=1e-6):
            print("intercept is verified.\n")
        else:
            print("error in caluclation\n")
            return
        
    def predict(self,x):
        if self.alpha is None:
            raise ValueError("Run solve() first.")
        x=np.array(x)
        if x.ndim==1:
            x=x.reshape(1,-1)
        return np.sign(x@self.w+self.b)

    def verify_sklearn(self):
        if self.alpha is None:
            raise ValueError("Run solve() first.")
        clf=SVC(kernel="linear",C=100.0,tol=1e-12)
        clf.fit(self.x, self.y.ravel())
        w_sk = clf.coef_.reshape(-1, 1)
        b_sk = clf.intercept_[0]
        alpha_signed_sk = np.zeros(self.m)
        alpha_signed_sk[clf.support_] = clf.dual_coef_.ravel()
        alpha_signed_ours = (self.alpha * self.y).ravel()
        print("========== SKLEARN VERIFICATION ==========")
        try:
            np.testing.assert_allclose(self.w, w_sk, atol=1e-5)
            print("Weight vector w verified")
        except AssertionError:
            print("Weight vector mismatch")
        print("ours    :", self.w.ravel())
        print("sklearn :", w_sk.ravel())
        try:
            np.testing.assert_allclose(self.b, b_sk, atol=1e-5)
            print("✓ Intercept b verified")
        except AssertionError:
            print("✗ Intercept mismatch")
        print("ours    :", self.b)
        print("sklearn :", b_sk)
        try:
            np.testing.assert_allclose(alpha_signed_ours,alpha_signed_sk,atol=1e-5)
            print(" Dual coefficients verified")
        except AssertionError:
            print(" Dual coefficients mismatch")
        print("ours    :", alpha_signed_ours)
        print("sklearn :", alpha_signed_sk)
        margin = 1 / np.linalg.norm(self.w)
        print("\nOur Model")
        print("w*      =", np.round(self.w.ravel(), 6))
        print("b*      =", round(float(self.b), 6))
        print("margin  =", round(float(margin), 6))
        print("\nSklearn Model")
        print("w*      =", np.round(w_sk.ravel(), 6))
        print("b*      =", round(float(b_sk), 6))
        print("\nSUCCESS: From-Scratch KKT Solver matches Scikit-Learn SVC")

# Example dataset:
# All three points are support vectors.

X = [
    [1.0, 2.0],   # +1
    [2.0, 1.0],   # +1
    [0.0, 0.0]    # -1
]

y = [1, 1, -1]

model = SVCaseLinearSVM(X, y)

# Solve KKT system
model.solve()

# Verify primal-dual identities
model.verifiy_duality()

# Compare against sklearn SVC
model.verify_sklearn()

print("\nRecovered Parameters")
print("alpha =", model.alpha.ravel())
print("w     =", model.w.ravel())
print("b     =", model.b)

# Predictions
test_points = [
    [3, 3],
    [0.2, 0.1],
    [1.5, 1.5]
]

pred = model.predict(test_points)

print("\nPredictions")
for point, label in zip(test_points, pred.ravel()):
    print(f"x = {point} -> class {int(label)}")