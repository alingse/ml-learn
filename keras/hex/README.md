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

  1. 根据上面 猜测的2，先训练一个二进制吧
     1. 准备数据
        根据输出个数`binlen`，确定各个的序列长度
        
        ```python
        def load_XY(binlen = 8):
            maxnum = eval('0b'+'1'*binlen)
            numlen = len(str(maxnum))

            count = maxnum + 1

            X_train = np.zeros((count,numlen),dtype=np.uint8)
            Y_train = np.zeros((count,binlen),dtype=np.uint8)
            for i in range(count):

                i_str = str(i).zfill(numlen)
                x_seq = np.array(map(int,i_str))
                i_bin = bin(i)[2:].zfill(binlen)
                y_seq = np.array(map(int,i_bin))

                X_train[i] = x_seq
                Y_train[i] = y_seq

        ```
        比如 13 --> x = [0,0,1,3] , y = [0,0,0,0,1,1,0,1] 
        左端用0补齐
        
     2. 选什么模型
        1. rnn
        
          因为是输入输出两个序列 ，如果按照时间来看，就是一维的数据依次输入，
          修改了`load_XY`(参考，[rnnbintrain.py](./rnnbintrain.py)) 使之形成序列
          但是这个没有训练出来，不懂为啥。（还要提高姿势水平才行，，，）
          
        2. cnn 
 
           测试了 binlen ＝ 1，2 时候，准备稳步推进训练
  2. 其他