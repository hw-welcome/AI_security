# ResNet 抗噪能力

刘路平 22151314



本实验测试预训练的ResNet在添加不同程度的高斯噪音的情况下的效果。

## 测试过程

- 使用预训练的ResNet
- 使用pytorch自带randn生成高斯噪音，根据一定比例alpha添加入Cifar10的测试集
- 使用ResNet预测结果并测试准确率

## 测试结果

![result](https://github.com/hw-welcome/AI_security/blob/master/homework1/data/ho1.png)

从测试结果来看，ResNet的抗噪能力比较一般，只需要加20%的高斯噪音就能使模型几乎失效。

为了给读者更好的直观，我们列举不同加噪比例alpha下的图片。

alpha = 0.1: ![](https://github.com/hw-welcome/AI_security/blob/master/homework1/data/01.png)

alpha = 0.2: ![](https://github.com/hw-welcome/AI_security/blob/master/homework1/data/02.png)

alpha = 0.3: ![](https://github.com/hw-welcome/AI_security/blob/master/homework1/data/03.png)



做的实在太简单了，就不放Github啦~



