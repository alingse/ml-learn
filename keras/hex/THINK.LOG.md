# hex －－ 进制转换，希望用CNN/RNN等训练一个一定范围内的hex函数－－实践、思考纪录
 
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
  2. 准备数据
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
     左端用0补齐(str.zfill)
        
3. 选什么模型
   1. rnn
     因为是输入输出两个序列 ，如果按照时间来看，就是一维的数据依次输入，这样很可能**序列不限长度估计也能推出来2进制**，类似做一个循环取模，辗转相除的东西出来－根据时间序列。
     针对性的修改了`load_XY`(参考，[rnnbintrain.py](./rnnbintrain.py)) 使之形成序列
     但是这个没有训练出来，LSTM。（还要提高姿势水平才行，，，）
          
    2. cnn 
      这个只能说，训练多少位的，多少位一下的才能用，
      还有一点就是，可能以后训练一个2位2进制转换，然后训练一个二进制加法器，
      会不会，只通过模型结合就可以处理不限长度的位了呢？（胡思乱想）
      

4. 那就cnn吧

  上模型吧
  
  ```python
    model.add(Dense(2*binlen,input_dim=numlen,activation=actives[0],init='uniform'))
    model.add(Dense(4*binlen,input_dim=2*binlen,activation=actives[1],init='normal'))
    model.add(Dense(2*binlen,input_dim=4*binlen,activation=actives[2]))
    model.add(Dense(binlen,input_dim=2*binlen,activation='hard_sigmoid'))

  ```
  测试了 binlen ＝ 1，2 时候，找了几个active，都能训练出结果来，
 
  ```python
        #['relu', 'sigmoid', 'linear']
        #['relu', 'sigmoid', 'hard_sigmoid']
        #['sigmoid', 'softmax', 'tanh']
        #['linear', 'softsign', 'tanh']
        #['linear', 'sigmoid', 'softsign']
        #['linear', 'softmax', 'tanh']
        #['linear', 'hard_sigmoid', 'softsign']
        #['hard_sigmoid', 'tanh', 'softsign']        
        #['softplus', 'sigmoid', 'softplus']
        #['softplus', 'softmax', 'tanh']
        #['softplus', 'softsign', 'relu']
        #['tanh', 'relu', 'relu']
        #['tanh', 'softmax', 'softmax']
        
        #['softplus', 'tanh', 'softsign']
        #['relu', 'softmax', 'tanh']
        #['relu', 'tanh', 'softsign']
        #['linear', 'tanh', 'linear']
        #['tanh', 'sigmoid', 'softsign']
  ```
  
  中间遇到的坑，数据量还是要复制多多的，打乱，
  
  然后训练的时候有一个考虑，，，就是 batch 的量跟数据重复度的问题，
  因为比如 binlen = 3 的时候，不重复的数据只有 2^3 = 8 个，复制200次，每次多少量够？
  会不会一次性所有数据都哪去训练，就很难收敛了，，，就是纠结要不要分批训练。
  
  再比如  binlen = 16，共65536组训练数据，一次训练多少合适呢？估计 10% ？
  感觉是 复制数要限制，batch 要合理的少一点，训练次数多无所谓，，猜想

5. 模型调整
  
  来来回回调试有多处调整，包括 loss 函数 最后采用了`binary_crossentropy`,感觉不错，
  
  优化函数开始用 `SGD` 后来用 `RMSprop`,感觉后者不错。
  
  三层单元，中间的个数也调整了，开始是 numlen-> numlen*numlen -> 4\*binlen -> 2\*binlen ->  bilen
  
  后来觉得，其实一上来多一点还好一点，后面慢慢去收敛，集中，只是猜的，需要理论
  
   改成了 numlen -> 2\*binlen -> 4\*binlen -> 2\*binlen -> binlen
   
   同时把train 函数 剥离出来，做了一些代码结构上的优化。
   
    
6. 目前训练结果
   
   针对 3-4-5-6－7-8 都训练出比较好的结果，模型结构也非常不一样。   
   上面 4 里面的表的参数，基本都有，但是在9这里卡住了。。
   
   训练量比较大，机子吃不住，准备安装GPU了，sad，还有一个学习速率lr 动态 的问题，希望前面跨步大，后期跨步笑，或者搞一下分批训练好了，这样或许收敛快一些。
   
7. 思考
 现在这个训练有何意义，除了能正面cnn可以做进制转换之外，就没有意义了，，因为没有预测呀，
   每次都是全部数据训练，，验证集都没有，准备砍掉20%的数据，然后，训练，然后看能不能预测。
   
   
   