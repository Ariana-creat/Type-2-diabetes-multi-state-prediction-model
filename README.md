# Type-2-diabetes-multi-state-prediction-model
Including the Cox proportional hazards model, random survival forest and gradient boosting machine, the Python version is 3.11.9.
Status definition: 0 indicates no diabetes, 2 indicates prediabetes, 1 indicates diabetes, and D indicates death.
Prediction task: D0: 0->1, 0->2, 0->D
                 D2: 2->0, 2->1, 2->D
                 D1: 1->D
