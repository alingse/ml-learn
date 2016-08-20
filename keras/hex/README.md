# hex - 进制转换，希望用CNN/RNN等训练一个一定范围内的hex函数
 
训练一个16进制转换器出来！！
 
##  猜测
1.  说是希望，是因为不知道能不能行，反正想要一个（`训练一个16进制转换器出来！！`）。

   一定范围内，是因为猜测进制转换跟累计加法器差不多吧

   所以 10位数的，可能 需要中间层多个输出吧，至少？

   输入数据、输出数据，最好要补齐统一位数

   类似这种：
   12 --> [0,0,0,0,0,1,2]  --> [0,0,0,C] --> C

2. 突然想到，为啥不是 10进制--> 二进制 --> 16进制

   第二个转换过程直接 4:1 输出就可以了，肯定很好训练吧，嗯
   
   所以输入的位数，直接关联到中间层的输出的单元数对吧，
   
   先来一个10进制到 二进制吧
   
   那感觉 好像每位上取余就可以了呢，就跟以前的那些单片机。加法器，减法器，等等，
   
   也许一层rnn，或者 N 层 CNN 就可以解决 N 位的问题了

## 过程

一些思考的过程参考[一些思考纪录：THINK.LOG.md](./THINK.LOG.md)

## model
 
  模型结果，基本就是四层结构，加上不同的activation
  
  ```python
    model.add(Dense(2*binlen,input_dim=numlen,activation=actives[0],init='uniform'))
    model.add(Dense(4*binlen,input_dim=2*binlen,activation=actives[1],init='normal'))
    model.add(Dense(2*binlen,input_dim=4*binlen,activation=actives[2]))
    model.add(Dense(binlen,input_dim=2*binlen,activation='hard_sigmoid'))

  ```
  
  
  输入输出个数是这样 `numlen -> 2*binlen -> 4*binlen -> 2*binlen -> binlen`
  
  常见的activations 组合是这样
  
   ```python
  #['softplus', 'tanh', 'softsign']
  #['relu', 'softmax', 'tanh']
  #['relu', 'tanh', 'softsign']
  #['linear', 'tanh', 'linear']
  #['tanh', 'sigmoid', 'softsign']
   ```
  
  现在训练出了3-4-5-6-7-8 的模型，可以很好的转换，
   
  转换方式是15 --> [1,5] 然后根据位数左端补齐

  变成[0,1,5] 输入模型，会得到 [0,0,0,0,1,1,1,1］ 即对应的0b1111
  

### TODO
#### 思考：

-  分割数据，做出有预测功能的2进制转换（好像不太可能，只要你是cnn就没办法超出自身的界限吧）
-  如果可以，有没有可能是像做出加法器减法器那样的模型，然后再级联，再做成hex`？
-  是否要用到rnn？（明明感觉序列的还不错呢）

#### real-todo：
- 调整训练策略，想办法快一点，不管是操作数据还是操作训练，多多学习
- 有可能还要改模型结构，看有没有办法做出序列化的东西

Thanks.