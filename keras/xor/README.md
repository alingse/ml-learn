# cnn XOR --- 用cnn来训练一个异或门

这里通过训练实现一个异或门model

异或就不用说了, 先准备的数据

```python
_data = np.array([[0,0],[0,1],[1,0],[1,1]])
_label = np.array([0,1,1,0])
```

## 调试经过

### v1
首先，根据 `A xor B = (not A and B) or (A and not B)`

猜测第一层需要4个输出，第二层2个输出即可（是因为输出有2种），

设计理念是 2 --> 4 -- > 2

所以大概最初设计是这个样子的： 

```python
model.add(Dense(4,input_dim=2,init='normal'))
model.add(Activation('relu'))
model.add(Dense(2,input_dim=4))
model.add(Activation('softmax'))
```

然后上训练数据，训练1000次，发现不行

```python
model.fit(data, label, batch_size=4,
                    nb_epoch=100,shuffle=True,
                    verbose=1,
                    validation_split=0.2)
```
各种调激活函数，测试，始终不行

 - 把层数增加，中间层输出个数增加到8、16、32，都不行
 - 改变优化的SGD，改变训练次数，都不行

最后发现最好的对原数据的预测结果也只是
`[0 1 1 1]`

突然一个激灵发现训练输出时显示数据只有3个

```bash
（某条训练log）
3/3 [==============================] - 0s - loss: 0.0067 - val_loss: 7.9206
```

才想到，原来只有4个数据，剩下1个拿来做测试。就是说最后一个数据，根本没有在训练里面，

### v2

立马加了8个进去，后来看了下`numpy`的`repeat`，改成repeat的了

```python
data = np.repeat(_data,20,axis=0)
label = np.repeat(_label,20,axis=0)
```
注意 axis=0，调一下就知道了

```
print(_data)
[[0 0]
 [0 1]
 [1 0]
 [1 1]]
print(data)
[[0 0]
 [0 0]
 [0 1]
 [0 1]
 [1 0]
 [1 0]
 [1 1]
 [1 1]]
```
返现还是不行，调到repeat 20 次还是不行，

看出来了，后面的`[1 1]`永远都是集中在后面，又好巧不巧前面的`validation_split=0.2`，刚好把这一块切出去了，

那就手动把原始数据打乱呗

```python
    #shuffle 避免repeat 数据集中重复
    index = list(range(len(label)))
    shuffle(index)
    data = data[index]
    label = label[index]
```

这下可以了，而且效果还行，训练了1000次，然后就开始调一些东西玩玩。

### vtest

1. 第一层的输出数跟训练次数

   发现 大一点会好一些，比如调到32 感觉会快一点，但是又想想，实际上4个才是最合理的吧，
   但是改成3、改成2 都可以，只是改成2的话，必须两层（这不废话嘛）
   
2. 数据复制次数
   
   复制次数少，确实不稳定，，，还是数据多多多才是正道

3. `validation_split ` 值调整

    在这个数据少的情况下,感觉训练量比交叉检验要重要，
    
    毕竟像[v1](#v1)情况下，根本没必要交叉验证。
    
    但是数据足够的情况下，合适的`validation_split`感觉可以快速调整收敛结果 

4. 各个activations 的测试

	目前官方的  activations 有如下几种，
	
	```python 
   actives = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
   ```
   来了一个内积，对每两对来测了同条件下5次训练的模型的评估score，代码：
   
   ```python
   actives = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
    result = []
    for A,B in product(actives,actives):
        s = 0
        for i in range(5):
            model = train(data,label,A,B)
            score = model.evaluate(data,label,batch_size=10,verbose=0)
            print(A,B,score)
            #test
            classes = model.predict_classes(_data)
            print(classes)
            s += score
        result.append((s,A,B))
    
    result = sorted(result,key=itemgetter(0))
   ```
   
   最后[结果](./result.sort)前几名和后几名是
   
   ```
softplus hard_sigmoid 0.00892226
tanh softmax 0.0127462
softplus softmax 0.0133235
tanh softplus 0.0149227
softsign softmax 0.0268711
softmax softmax 0.0287166
.................
.................
relu relu 37.9949
tanh relu 38.0169
softplus relu 39.513
linear relu 40.0097
softsign relu 40.0756
linear tanh 40.2074
sigmoid relu 40.2074
hard_sigmoid relu 40.2513
relu tanh 40.2952
   ```
   最后选了一个 `tanh softmax`
   
## 一些体会

  - 数据是**第一重要**的，越多越好，训练不够很难预测，因为模型要见多识广呀，
      
    训练数据不要集中，要分散一些，
  - 层数跟输出层还是要根据问题复杂度来设，尽量不能少，也别多
  - 参数配置要思考，有些前后是有联系的
  - 有些设置是铁定不会收敛的，所以多尝试不同参数
  - numpy是个好东西，各种api和切片要小心。

## TODO
  - 想玩GUP和更复杂的问题
  - 多学内置参数，及其适用范围，就是看文档
  - 学numpy的更多细节，就是看文档