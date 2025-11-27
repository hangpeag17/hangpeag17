本项目构建了一个用于 信贷违约预测（Credit Default Prediction） 的机器学习系统，结合：
>>LightGBM（树模型）
>>DAE-ResNet 神经网络（深度结构化数据模型）
>>SAINT Transformer（Self-Supervised Tabular Transformer）
>>OOF / stacking / 特征融合
目标是在金融结构化数据场景中获得 稳定、可解释、可泛化的高 AUC(0.79~0.80)预测性能。
>在进行这些模型之前需要先对数据文件进行预处理与特征工程可见data_EDA.ipynb,在这里我是先对每个文件的非数值型列进行了独热编码之后再去聚合,最后再通过merge函数进行拼接得到了我们最终的train_final与test_final(one-hot版本)
>>


