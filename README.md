# Dynamic-Weighted-Stacking-Ensemble-Net
#READMEcn.md
1.以数据集命名的文件，就是使用该数据集进行训练

2.序号以m.n来排序，
①m为1时指各个单模型训练，n从1到3，分别是是单模型NeZha、XLNet、ERNIE
②m为2时指用各种融合方法训练，n从1到2分别是动态融合器（DynamicFusionNet）、动态加权堆叠融合（DynamicWeightedStackingNet）
器（DynamicWeightedStackingNet）
③m为3时指用各种融合方法多折交叉验证训练，n从1到2和上面m为2的情况一样

3.融合网络介绍
①动态融合网络（DynamicFusionNet）
动态融合器是一种基于动态加权融合的模型集成技术，专注于通过精细调整各模型的权重来优化分类任务的结果。该融合器将多个强大模型（如NeZha, XLNet, ERNIE）
的输出结果动态加权，以确保在每个任务中的表现最优。与传统的简单平均或固定权重不同，动态融合器使用贝叶斯优化算法动态调整模型的权重，从而在不同任务和数据
集上保持高效的预测性能。该方法适用于各种分类任务，尤其在处理多样化和复杂数据时表现出色。

②动态加权堆叠融合网络（DynamicWeightedStackingNet）
动态堆叠融合器是一个结合了动态加权和Stacking集成的复杂模型集成方案。在该系统中，多个强大的模型（如NeZha, XLNet, ERNIE）独立地处理输入数据，并通
过贝叶斯优化动态调整各模型的权重，生成加权融合的初步结果。这些加权结果然后通过Stacking集成进一步处理，利用次级模型（通常是一个简单的回归或分类模型）
来学习和优化最终的分类结果。动态堆叠融合器利用了不同模型的互补性，显著提升了整体模型的预测性能和泛化能力。

建议使用RTX4090及以上显卡，若显卡内存不足可适当减少batch_size的大小

#READMEen.md
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
