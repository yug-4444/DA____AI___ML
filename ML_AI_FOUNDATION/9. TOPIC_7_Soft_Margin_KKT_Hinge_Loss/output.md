PS D:\Data_Structure__AND__AI___VSCODE\DA____AI> python -u "d:\Data_Structure__AND__AI___VSCODE\DA____AI\ML_AI_FOUNDATION\9. TOPIC_7_Soft_Margin_KKT_Hinge_Loss\Soft_Margin_KKT_Hinge_Loss_primal_dual_system.py"
Dataset Shape : (308, 2)
Labels Shape  : (308, 1)

======================================================================
METHOD : BGD
======================================================================

PRIMAL
Train Accuracy : 0.948051948051948
Test Accuracy  : 0.948051948051948

DUAL
Train Accuracy : 0.948051948051948
Test Accuracy  : 0.948051948051948

COMPARISON
Weight Difference : 0.006187532156923975
Bias Difference : [0.01864705]
Duality Gap : 0.12082092477205464

======================================================================
METHOD : SGD
======================================================================

PRIMAL
Train Accuracy : 0.948051948051948
Test Accuracy  : 0.948051948051948

DUAL
Train Accuracy : 0.9523809523809523
Test Accuracy  : 0.948051948051948

COMPARISON
Weight Difference : 0.4097272928217186
Bias Difference : 0.11381557688630092
Duality Gap : 64.0645864550278

======================================================================
METHOD : MINI-BGD
======================================================================

PRIMAL
Train Accuracy : 0.948051948051948
Test Accuracy  : 0.948051948051948

DUAL
Train Accuracy : 0.948051948051948
Test Accuracy  : 0.948051948051948

COMPARISON
Weight Difference : 0.00941253776581752
Bias Difference : [0.02118781]
Duality Gap : 0.12039980717975851

======================================================================
PREDICTION EXAMPLES
======================================================================
Point [3 3] --> Predicted Class 1
Point [-3 -3] --> Predicted Class -1
Point [0 0] --> Predicted Class -1
Point [ 2 -2] --> Predicted Class 1
Point [-2  2] --> Predicted Class -1
Point [1 1] --> Predicted Class 1
Point [-1 -1] --> Predicted Class -1