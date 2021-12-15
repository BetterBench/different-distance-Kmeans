# 【机器学习】Python详细实现基于欧式、余弦、切比雪夫、曼哈顿和杰拉德距离的Kmeans聚类

# 1 算法过程

（1）随机选取K个簇中心点

（2）通过计算每个样本与每个簇中心点的距离，选择距离最小的簇心，将样本归类到该簇心的簇中

![[公式]](https://www.zhihu.com/equation?tex=min+%5Csum%5E%7BK%7D_%7Bi%3D1%7D+%5Csum+_%7Bx+%5Cin+C_%7Bi%7D%7D+dist%28c_%7Bi%7D%2Cx%29%5E2)

这里距离可以使用欧几里得距离（Euclidean Distance）、余弦距离（Cosine Distance）、切比雪夫距离（Chebyshew Distance）、曼哈顿距离（Manhattan Distance）或杰拉德距离（Jaccard Distance），计算距离之前需要先对特征值进行标准化。

3、在已经初次分配的簇中，计算该簇中所有向量的均值，作为该的簇中心点

4、重复步骤2和3来进行一定数量的迭代，直到簇中心点在迭代之间变化不大



评价指标

> 轮廓系数（Silhouette Coefficient），是聚类效果好坏的一种评价方式。   轮廓系数的值是介于 [-1,1] ，越趋近于1代表内聚度和分离度都相对较优。



## 2 实验结果

（1）曼哈顿距离

轮廓系数：0.7333423486262539

![](https://img-blog.csdnimg.cn/2dff78176d034db7bcf1e37de79ed75f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmV0dGVyIEJlbmNo,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)

（2）切比雪夫距离

轮廓系数0.7333423486262539



![](https://img-blog.csdnimg.cn/e263336a358b44febf1420b7947998e0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmV0dGVyIEJlbmNo,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)

（3）欧式距离

轮廓系数：0.7333423486262539

![](https://img-blog.csdnimg.cn/6af3f6836c144948ab73d2e158bbecd0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmV0dGVyIEJlbmNo,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)