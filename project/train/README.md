We have 4 main scripts we used during the training procedure.
They are labeled accordingly to their order of use.
Our training method inspired from "transfer learning" idea. We first train over entire input power range, finally we fine tune our model for each specific input power and creat multiple models. 

* [STEP 1](https://github.com/skrimpon/nonlin/blob/main/project/train/step1_train.py)
>This is the initial training step, where we train our model using every input power value. However, we still feed only one specific power value at a time to our network, we make pass over all the power range multiple times and record our model evaluation. 

* [STEP 2](https://github.com/skrimpon/nonlin/blob/main/project/train/step2_train_optimize.py)
>In this script we train one model per each input power value, we use the power values near the target power value as well.


* [STEP 3](https://github.com/skrimpon/nonlin/blob/main/project/train/step3_model_clean.py)
>This script filters out the final models and moves them to a clean folder for evaluation.

* [STEP 4](https://github.com/skrimpon/nonlin/blob/main/project/train/step4_result_plotter.py)
>This script evaluates our final models on the test data of 40,000 samples. This script will produce the following figure.
![eval](https://raw.githubusercontent.com/skrimpon/nonlin/main/performance_eval.png)
