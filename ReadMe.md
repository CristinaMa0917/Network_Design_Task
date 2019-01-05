### 深度学习网络搭建Tasks小结
*考核给出数据和任务说明，根据特定的任务选取对应的网络结构，搭建出在限制条件下performance优越的网络。考核对网络结构的理解，以及熟练运用tensorflow，keras，pytorch的能力。经过实践对比，pytorch更易搭建和debug，训练速度和精确度略优于tf，因此git的代码是基于pytorch平台的。*

1. ##### task1 
    - Describe：找出一个序列中的最大值的起始位置
    - Data：32000个长度为10的样本，每个样本包括10个0-9的数
    - Notes: 两层MLP
    - Constraints: batch = 32, lr=0.01, optimizer=Adam，epoch=1
2. ##### task2 
    - Describe：找出一个序列中连续三个数相加最大的子序列的起始位置
    - Data：32000个长度为10的样本，每个样本包括10个0-9的数
    - Notes:两层卷积
    - Constraints: 
        1. inductive bias尽量小：测试集的准确度达到98%
        2. 训练准确度达到96%所需的batches小于1000
3. ##### task3 
    - Describe：输入一个序列和一个标量N，找出该序列中连续N个数相加最大的子序列的起始位置
    - Data：32000个长度为10的样本【passage】,32000个长度为1的样本【query】
    - Notes: 一层LTSTM，一层全链接 
    - Constraints: 
        1. batch = 32, lr=0.01, optimizer=Adam，epoch=1
        2. 为避免直接利用数字信息，将sequence和query统一embed到8维空间作为输入
        3. 因为建模随机性，只要5次有一次能超过指标即可
        4. 最后100个batch的平均准确率大于91%（bonus指标：大于95%）（5）100个batch的平均准确率稳定超过80%所需要的batch小于1100
    
4. ##### task4 
    - Describe训练：输入一个序列和一个标量N，连续N个数相加最大的子序列的起始位置为P1，同一P1的序列认为是一类
预测：输入一个序列和一个标量N，连续N个数相加最大的子序列的起始位置为P2，其中P1和P2的并集为空；将P2集合分成P2_1(target)和P2_2(distractor)，在P2_1=i中，选取两个（sequence，query），一个作为anchor，一个作为target混入distractor中，求target与anchor排名rank 1的平均准确度
    -  Constraints:
       1. 完成一个平均准确度超过50%的baseline 
       2. 根据错误寻找一个优化点（optmization strategy方面的或是结构方面的优化点均可），完成并在baseline上有non-trivial的提升 
       3.可跑多个epochs，可调参，注意regularization 
       4. evaluation脚本：python3 evaluate.py embeddings.csv task4_test_label.csv，其中embeddings无header，无index，每行是一个embedding
    
5. ##### task5
    - Describe：输入为两个长度为N的序列，序列1中任意一个数a和序列2中任意一个数b，如果|a-b|=15，则认为a和b相似。
两个序列中所有相似对（a，b）的个数是偶数时，label=0；奇数时，label=1
    - Hints：设计一个alignment模块
    - Constraints:
        1. 结构设计，不用调参，固定batch = 32, lr=0.01, optimizer=Adam，epoch=1 
        2. 为避免直接利用数字信息，将序列中的数字映射到8维空间作为输入
        3. 不设定硬性考察指标，请大家自行探索最优结构，最后进行测试集排行

6. ##### task6
    - Describe：一个序列由1和0组成，判断序列中1的个数是奇数还是偶数
    - Constraints：
        1. 结构设计，不用调参，固定batch = 32, lr=0.01, optimizer=Adam，epoch=1
        2. 为避免直接利用数字信息，将序列中的数字映射到8维空间作为输入
        3. 鼓励尝试各种结构，不过至少一种结构是基于RNN的
        4. 测试集accuracy达到100%
        
7. ##### task7
    - Describe：设计网络结构，预测二维物体的对称性和空隙率
    - Data：二值化图片
    - Notes: 4层卷积，3层dense
    
8. ##### task8 
    - Describe：设计网络结构，反转一个变长序列（最大长度N=20），即246910000反转为196420000，其中1-9为需要反转的有效字符，0为补位字符
    - Data：1. 序列长度很长时，如何记住前序信息 2. output structure是序列的优化方法

    - Notes: 三层GRU加两层全链接，准确度达87%
    - Constraints：
        1. 结构设计，不用调参，固定batch=32， lr=0.02, optimizer=Adam, epoch=1
        2. 将序列中的数字映射到8维空间作为输入 
        3. 禁止直接将input与output层相连

   - 解决方案： 三层GRU（前两层双向）+ 一层Dense 
   - Train上的准确率 （final 100 ： 0.907 ）
   - Test 上的准确率 0.87

