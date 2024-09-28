1.Files named after datasets are those used for training with the corresponding dataset.

2.Files are sorted by the numbering scheme m.n:
①When m is 1, it refers to individual model training. The value of n ranges from 1 to 3, corresponding to the single
models NeZha, XLNet, and ERNIE, respectively.
②When m is 2, it refers to training with various fusion methods. The value of n ranges from 1 to 2, representing Dynamic
Fusion Network (DynamicFusionNet) and Dynamic Weighted Stacking Network (DynamicWeightedStackingNet), respectively.
③When m is 3, it refers to training with various fusion methods using cross-validation. The value of n ranges from 1 to
2, and it corresponds to the same fusion methods as when m is 2.

3.Fusion methods introduction:
①Dynamic Fusion Network (DFNet):
The Dynamic Fusion Network is a model ensemble technique based on dynamic weighted fusion, focused on optimizing
classification task results by finely tuning the weights of each model. This fusion network dynamically adjusts the weights of the output results from multiple powerful models (such as NeZha, XLNet, ERNIE) to ensure optimal performance for each task. Unlike traditional simple averaging or fixed weights, the Dynamic Fusion Network uses Bayesian optimization to dynamically adjust model weights, maintaining efficient predictive performance across various tasks and datasets. This method is particularly effective for diverse and complex data classification tasks.

②Dynamic Weighted Stacking Network (DWSN):
The Dynamic Weighted Stacking Network is a complex model ensemble approach that combines dynamic weighting with Stacking
 integration. In this system, multiple powerful models (such as NeZha, XLNet, ERNIE) independently process input data, and their weights are dynamically adjusted using Bayesian optimization to generate initial weighted fusion results. These weighted results are then further processed through Stacking integration, where a secondary model (usually a simple regression or classification model) is used to learn and optimize the final classification outcome. The Dynamic Weighted Stacking Network leverages the complementary strengths of different models, significantly enhancing overall predictive performance and generalization ability.

It is recommended to use an RTX 4090 or higher GPU. If the GPU memory is insufficient, you can reduce the batch size
accordingly.