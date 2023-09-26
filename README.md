最近在做光谱信号的处理，需要用到网络，加上太久没用pytroch了，索性就参考了几个模板，使用DeepInverse（图像）网络结构，写了一个pytroch处理1维信号的模板。

> 此模板的存放仓库：[GitHub仓库地址](https://github.com/similing-scholar/1D_DeepInverse_pytorch.git)、[gitee仓库地址](https://gitee.com/similing-scholar/1D_DeepInverse_pytorch.git)

>参考的模板：1. [**torchkeras**](https://github.com/lyhue1991/torchkeras) 模仿keras接口写的自定义Model类，开发成包； 2.[**PyTorch_Tutorial**](https://github.com/TingsongYu/PyTorch_Tutorial/tree/master/Data) 关于pytorch构建模型每个部分非常通俗且详细的讲解以及code，有pdf；     3.[**Pytorch-UNet**](https://github.com/milesial/Pytorch-UNet) 清晰的文件上下级关系，主目录下train.py与predict.py方式运行；