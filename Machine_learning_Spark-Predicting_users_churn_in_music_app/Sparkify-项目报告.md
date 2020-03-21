## 数据挖掘工程师直通班
---

# 项目名称：分析Sparkify软件用户数据，预测客户流失
---

##### Travis Choi，Udacity
2020年2月18日

# 一，定义
---

## 项目背景
---

- 本项目数据集来自于Sparkify，Sparkify是一款听歌软件，拥有海量的用户，用户分为**`会员用户`**(每月付费，可免费听歌)、**`普通用户`**(不付费，有广告植入)。

- 每次用户与产品互动时(如听歌、登录登出、点赞、看广告、服务升级降级等)，均会产生数据记录，本项目的数据集即为这些记录的汇总。

- 公司通过会员用户的付费盈利，通过分析数据集，观察和总结用户行为，进而挖掘隐藏在数据背后的信息，以便推出相关决策，维持和增加会员用户的数量，最终增加公司盈利水平。

## 项目目标
---

- 观察和分析用户数据，关注用户注销的行为(用户流失，取关或注销Sparkify，不再产生用户互动数据记录，不再有转化为付费会员的可能)，并从其行为数据信息中提取相关特征，并利用机器学习模型进行拟合，进而选用表现较优的模型，对后期潜在的取关行为进行预测。

## 主要应用
---
本项目在spark编程语言环境下执行，辅以python和SQL。本项目为**`二元分类`**问题，在监督学习中，选择了**`决策树、逻辑回归、随机森林、支持向量机、提升树模型`**。此外，项目数据0—1分类并非比例相当，实际上发现**分类1(用户流失)的比例较低(52/225)，所以在评估指标的选择时，更偏向于兼顾查准率Precision和查全率Recall两者的`F1Sore`指标**，辅以准确率Accuracy。

- 标准库
    - **`spark:`** pyspark.sql, pyspark.ml
    - **`python:`** pandas, numpy, matplotlib
    - **`SQL`**
    
- 评估指标
    - **`准确率`**：Accuracy
    - **`F1Score`**
    
## 数据字典
---
- artist:艺术家、歌手名
- auth: 登录状态、属性
- firstName: 用户名
- gender: 性别
- itemInSession: 存储位置
- lastName: 用户姓
- length: 活动时长
- level: 用户等级
- location: 地域
- method: 获取方式
- page: 操作页面
- registration: 注册状态
- sessionId: 存储Id
- song: 歌曲名
- status: 网络状态码
- ts: 时间戳
- userAgent: 用户代理、终端系统
- userId: 用户Id

    
# 二，数据评估和清洗
---
### 缺失值
- 通过数据评估，发现artist、firstName、gender、lastName、length、location、registration、song、userAgent列存在**`Nan值`**，但也发现page列并无Nan值，为了最大程度跟踪用户行为，暂且不能随意删除这部分数据。
- 另外，发现**`userId列存在名称为空`**的情况，此部分对后期分析无意义。删除这些空项的所在行后，通过再次check数据列，发现userId列存在名称为空的行已完全删除，此外列artist、length、song仍存在Nan值，为用户切换page时出现的正常Nan值，但仍有效的记录了用户行为，需要保留这部分数据。

### 格式转换
- 将时间戳转换为时间格式，查看用户注册时间和活动时间，并进一步计算用户**`注册天数`**(注册至数据数据截止日的天数，或注册至用户流失日的天数)


### 定义客户流失
- 根据观察发现伴随**`page='Cancel'`**出现的**`page='Cancellation Confirmation'`**为用户最后一条活动信息，之后彻底取消关注Sparkify听歌软件，该客户确认流失。按照用户行为是否存在Cancellation Confirmation事件，做0-1分类，作为后期生成机器学习的label项做准备。


### column整理
- **`level`**：发现同一个**`userId`**会对应多个**`level`**,说明有的用户曾经付费过，后来仍然流失了，后期将该这部分用户统一规划为paid。
- **`page`**：
	- 实例观察，发现**`Sumbit Upgrade`**为主动付费升级标志页面，而**`Sumbit Downgrade`**为主动降级标志页面，当两者出现时，level的状态才会发生改变，更具分析意义。
	- **`Settings`**和**`Save Setting并`**不同时出现，前者数量明显多，或为用户不经意打开或者浏览app功能，而后者表示用户保存设置，此交互行为在一定意义上体现出客户对app使用的深入和个性化导入，从而更具分析意义。

	- 此外**`About、Home、Logout、Error`**均为日常操作常规界面，其数据记录的情况类似于**`Setting`**，相对不具有分析意义，待后续EDA核验。
	- 其他项均代表客户常规操作（即**`Add Friend、Add to Playlist、NextSong、Roll Advert`**等），此时通过聚合，生成代表各项操作计数的spark DataFrame，并最终jion至同一个DataFrame中，便于后期EDA和特征工程，其各项概览如下：

	> Row(userId='100010', gender='F', avg_length=243.42144478537818, alive_ts=4807612000, alive_days=55.64365740740741, paid=0, puts=313.0, gets=68.0, add friends=4.0, add to playlists=7.0, cancellation confirmations=0.0, errors=0.0, helps=2.0, nextsongs=275.0, roll adverts=52.0, save settingss=0.0, submit downgrades=0.0, submit upgrades=0.0, thumbs downs=5.0, thumbs ups=17.0, label=0)


# 三，数据分析和可视化
---
- 本章节主要对用户的**`属性和行为`**进行可视化分析，且为了方便绘图，事先将其转换为pandas的DataFrame格式，并定义了相关函数，用于柱状图绘制。
- 本章目的在于观察数据整体情况，并重点关注未流失和流失客户群体的行为比较，从而进一步选取特征，为后期监督学习做铺垫。
- 需特别注意的是，根据常识可知，因流失用户在流失后不再产生数据，所以未流失用户的平均数据量一定大于流失用户的平均数据量，所以在定义用户行为时，**`务必需着重考虑客户行为的频率，而不仅仅关注行为的总数。`**
- 用户行为频率 = 用户行为总数/注册天数

## 探索性可视化
### 用户听歌数
- 按照0-23h划分，绘制散点图，查看用户在一天中的活动情况，可发现用户多在晚上和凌晨听歌较多：

![The songs count in hours](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_104_0.png)

### **`注：以下柱状图中，label值为1表示流失用户，label值为0表示未流失用户`**

### 性别(gender)
- 可以看出，相对而言男性客户流失比例较高,可作为区分特征之一。

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_109_0.png)

### 注册天数(alive_days)
- 由于未流失用户一直在使用app，如期比流失用户产生更多的数据量、具有更长的注册时间：
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_112_1.png)

### 听歌时长(length)
- 经比较发现，流失客户和未流失客户的平均听歌时长基本相同，此特征无法对两列用户进行有效区别，不能作为区分特征之一。
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_115_1.png)

### 用户等级(level)
- 付费客户中，未流失和流失的用户比例：
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_118_1.png)
- 流失用户中，免费与付费客户的比例：
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_119_1.png)

通过对比发现，在流失和未流失两类客户中，付费会员的比例基本相当。而且付费行为并未有效阻止客户流失，推测在软件的用户体验上还有待进一步改善。要重点关注付费用户的流失情况，并作为区分特征之一。

### 获取方式(method)
通过观察发现，在puts和gets的平均数量上，未流失用户明显多于流失用户的，或与未流失用户的累计app使用时间更长有关。

- PUT数量比较：

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_122_1.png)

- GET数量比较：

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_123_1.png)

#### 为了查看真实情况，需进一步计算和观察用户行为频率：
- PUT频率比较：

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_126_1.png)

- GET频率比较：

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_127_1.png)

通过进一步比较发现，用户在puts和gets频率的比较上，仅后者有较明显的区别，可以将其作为用户区分的特征之一。

### 用户操作页面
查看page列的总体情况，对比未流失和流失客户的行为总数。
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_135_0.png)
因NextSong一项的数量明显多，影响其他项的对比，所以将其去掉后重新绘图：

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_137_0.png)

- 以上图表展示了用户查看页面的平均次数，并**`未考虑用户注册至今/流失的时长(注册天数alive_days)影响`**，即并未考虑用户行为的频率问题，有待进一步探索。
- 上图如预期所示，大部分page项在数量统计上，未流失>流失，**`需特别关注其中未流失≈流失，甚至未流失<流失的page项`**，如Roll Advert、Upgrade。

### 添加好友
通过观察发现，未流失和流失用户添加好友的频率基本一致，后者稍微较多，可以作为区分特征之一。

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_141_1.png)

### 添加至播放列表
通过观察发现，未流失和流失用户添加歌曲至列表的频率，后者较多，可以作为区分特征之一。

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_144_1.png)


### 页面出错
通过观察发现，未流失和流失用户网络错误的频率前者较多。但从逻辑上讲，出错多必定影响用户体验，不适合作为区分特征之一。

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_147_1.png)


### 查看帮助
通过观察发现，未流失和流失用户查看帮助的频率后者较多，或是因为流失客户对app的现有功能不满，有较多的求助情况，可以作为区分特征之一。

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_150_1.png)

### 广告
通过观察发现，流失用户观看广告较多，频率明显高，此特征应重点关注，作为区分特征之一。

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_153_1.png)


### 保存设置
通过观察发现，未流失和流失用户虽然前者Setting的次数多，但后者频率较高。从逻辑上讲，用户的Setting动作，在一定程度上讲，是其对app依赖和不满的综合体现，应将其作为区别特征之一

![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_156_1.png)


### 升级降级




#### 主动Upgrade的频率：
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_159_1.png)

- 通过观察发现，未流失和流失用户，后者升级频率较高。从逻辑上讲，用户的升级动作，是其付费或参与活动体现，升级后应具备更好的用户体验，然而依然会流失，侧面说明app的体验有待改善。

#### 主动Downgrade的频率：
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_160_1.png)

- 未流失和流失用户降级的频率，前者稍多,理论上降级务必影响用户体验，但实际并非如此。
- 结合升降级来看，流失的客户主动升级多，主动降级少，可以推测该部分用户注重app体验，其更倾向通过升级来提高服务，而非通过降级来提高性价比，但最终仍然流失的原因，或为通过不断升级，其发现app体验并未达到其预期，导致最终放弃和注销产品。应将二者作为区别特征之一。

### 踩和赞

#### '踩'的频率：
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_163_1.png)

明显看出，流失用户踩的频率高，应作为特征之一。

#### '赞'的频率：
![png](https://github.com/TravisChoi07/Machine_learning_projects/blob/master/images/output_165_1.png)

通过观察发现，未流失和流失用户点赞的频率基本一致，前者稍微较多，不能作为区分特征。

## 数据保存
通过EDA筛选出可以作为未流失和流失用户两者特征区别的column，包括：**`'userId','gender','paid','gets_freq', 'addfrd_freq', 'adlist_freq','help_freq', 'ads_freq', 'set_freq', 'upgrd_freq', 'dwngrd_freq','thmdwn_freq', 'label'`**，共13列，将其进行筛选出并导出为**`json文件`**。

# 四，技术和建模
---
## 特征工程

- 数据加载：将json文件导入，并建立spark环境。
- 格式转换：将用于定义分类类型的非数字项转换为数字形式。
- 向量转换：将所有特征放入一列，并进行向量转换。
- 标准化：数据标准化处理，为监督学习做准备。

## 建模
- 主要思路：采用用户行为数据来定义特征(features)。通过是否取消关注，定义标签(label)。
- 主要步骤：
	- 拆分数据集：按照**`6：2：2`**的比例，将数据集拆为**`train，validation，test`**三个数据集。
	- 天真预测器：基准测试，即假设全部用户全是/全不是潜在流失用户时，计算test集的基准准确率和F1Score，作为后期监督学习模型评估的基准。实际基准为accuracy：0.721，F1Score：0.604。
	- 监督学习：利用交叉验证和调参进行模型训练，模型包括**`决策树、逻辑回归、随机森林、支持向量机、提升树模型`**，待模型获得最优参数后，查看模型表现。
## 模型改进

- 建模在最初期并未顺利，模型的训练结果不达标，F1Score均低于65%，刚刚优于基准水平，甚至更差。
- 反思结果不达标的原因可能有二：**`模型参数调优问题、features数量和选择问题。`**
- 最终将**`features数量`**由7个增加到11个，增加grid中的参数数量，进行更全面的训练后，最终获得以上结果。
	
# 五，结果和结论
---
## 监督学习模型及结果：
- 决策树：DecisionTreeClassifier()，无参数输入
	- accuracy：验证集0.755，测试集0.744
	- F1Score： 训练集0.729，验证集0.737，测试集0.720
- 逻辑回归：LogisticRegression(maxIter=6 regParam=0.0)，其中maxiter表示最大迭代数量，当其取值为6时获得最优表现，regParam表示正则化的参数，取值为0时表现最优。
	- accuracy：验证集0.849，测试集0.767
	- F1Score： 训练集0.700，验证集0.821，测试集0.701
- 随机森林：RandomForestClassifier(numTrees=10)，numTrees表示应用决策树的数量，取值为8时，表现最优。
	- accuracy：验证集0.774，测试集0.791
	- F1Score： 训练集0.700，验证集0.708，测试集0.758
- 支持向量机：LinearSVC(maxIter=4，regParam=0.1)，其中maxiter表示最大迭代数量，当其取值为4时获得最优表现，regParam表示正则化的参数，取值为0时表现最优。
	- accuracy：验证集0.811，测试集0.721
	- F1Score： 训练集0.676，验证集0.727，测试集0.604
- 提升树：GBTClassifier(maxIter=2)，其中maxiter表示最大迭代数量，取值为1或2时获得最优表现。
	- accuracy：验证集0.755，测试集0.744
	- F1Score： 训练集0.729，验证集0.737，测试集0.720

## 总结

### 建模总结
逻辑回归>决策树=提升树>随机森林>支持向量机

- 在监督学习中，通过各模型的性能比较，发现模型的准确率accuracy范围在0.721~0.849之间，均高于基准accuracy (0.721)，模型的F1Score范围在0.604~0.821之间，均高于基准F1Score (0.604)。

- 以F1Score为主，accuracy为辅，综合衡量5个模型，从数据值及稳定性上来讲，**`逻辑回归`**模型表现最优，其次是**`决策树`**和**`提升树`**模型，此二者结果完全相同，且表现十分稳定，此外**`随机森林`**模型的表现也不差。通过这4个模型对未来数据做出预测，都会有相对较好的预期效果。
- 在监督学习的模型中，**`决策树、随机森林、提升树`**都属于树模型，而**`随机森林`**的效果稍微差一点，分析应该是由于在**`随机森林`**中，多个树分别对模型的部分features进行fit，最终以投票的形式决定分类，其大费周折却并不如前面2种树模型对本数据的拟合更贴合。其次，也可能是由于数据量不够大，导致结果上的误差，倘若增加fit的数据规模，其表现或有所改善，甚至超过前面2种树模型。
- **`提升树`**模型也是高效分类的模型之一，值得一提的是，在此建模的过程中，发现其树的数量为1或2时表现最好，且与无参数的**`决策树`**模型的表现完全一致，树的数量越多，模型的表现反而越差，说明提升树模型迭代和加权汇总的分类形式可能过于复杂，而本项目数据本身对属于某类的概率、权重并不敏感，并不需要如此精确的计算。
- **`逻辑回归`**模型的表现最优。分析其原因为，逻辑回归是在线性回归基础上加入**`sigmoid函数`**，换句话说，倘若数据特征的综合表现、整体倾向于0和1的哪一边，哪怕只是稍微倾向，即为其分归该类。所以在本项目的模型fit中，“精于计算”的**`提升树`**模型和**`支持向量机`**模型表现并不理想，尤其后者表现最差，其在测试集中的测试结果与基准值完全相等。


### 项目回顾
- 本项目先后经过了**`数据加载、评估和清洗、可视化探索分析、特征工程、建模和优化`**几个步骤，并在不断修改的过程中，逐渐获得更符合预期的结果。
- 在项目进行过程中，思路清晰。但实现的过程中**`充满挑战`**，如：
	- 如在分析feature选择的取舍时，如何**`综合考虑选择逻辑、模型训练效率问题`**；
	- 在可视化数据分析过程中，何时使用spark进行数据聚合，何时将数据转换为pandas的DataFrame格式，从而更方便高效的输出可视化图形；
	- 如在数据分析的过程中，如何将重复的功能变为**`自定义函数`**，符合“**`DRY`**”原则，用更少的代码，更方便快捷的达到预期效果；
	- 如在严谨性方面，考虑到未流失和流失用户在**`注册天数`**上的客观不一致性，会导致前者数据量明显大，而不适合直接以各feature数据量的多少作为训练模型的依据，应加以处理后获得用户的行为频率，以此作为features才更符合实际情况，由处理过的features训练的模型，才能更适合项目落地。

### 局限性

- 一是客观**`数据集规模`**较小，难免会影响模型训练的准确性；
- 二是特征工程和监督学习或有更好的处理方式，比如采用其他机器学习模型进行建模；
- 三是训练结果，F1Score在70~80%并非十分理想，后续进一步挖掘数据特征(如地域、艺术家)，仍有继续提高的可能。

### 参考文献/网址
- [Report template](https://github.com/udacity/machine-learning/blob/master/projects/capstone/capstone_report_template.md)
- [Spark 2.4.5](https://spark.apache.org/docs/latest/sql-getting-started.html)
- [JDK install](https://blog.csdn.net/weixin_45281949/article/details/104247038)
- [PySpark Classification and regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier)
- [pyspark.ml package](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.DecisionTreeClassifier)
- [pandas.DataFrame.to_json](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html)
- [pyspark.dataframe聚合:agg](https://blog.csdn.net/weixin_42864239/article/details/94456765)
- [关于pyspark.sql中udf返回类型](https://blog.csdn.net/dkjkls/article/details/90742439)
- [pyspark.dataframe：join操作](https://www.jianshu.com/p/310bc198ea19)
- [HTTP常见状态码200、404](https://www.cnblogs.com/starof/p/5035119.html)
- [pyspark.dataframe 中更改str为float](https://cloud.tencent.com/developer/ask/176110)
- [时间戳转换为时间格式](https://blog.csdn.net/LeonTom/article/details/83586469)
- [matplotlib.pyplot 双柱状图](https://www.jianshu.com/p/8c4a29a0cfc2)
