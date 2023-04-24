## 更新日志
### <font>2023/3/19 更新</font>
+ 重构项目中以'server.py'与'client.py'为主体的联邦学习模块
+ 将联邦学习模块整合入服务器路由，现可通过'/fl_server'与'/fl_client'路由对服务器与客户端进行分别访问
+ 增加'logs'目录用以存放联邦学习产生的日志，联邦学习关闭后可自动删除对应日志 (计划之后各模块日志都附属于此目录下)
+ ~~<font color="red">仍需讨论关于界面表现与联邦学习进入方式的问题</font>~~<font color="green">基本完成</font>
+ <font color="red">无法关闭联邦学习服务器，现用守护进程保证其在主进程关闭时可释放资源，但无法即时关闭此服务器</font>
+ <font color="red">无法将联邦学习过程中产生的tensorflow日志重定位至标准日志文件，会导致前端无法清晰地显示训练过程</font>
+ <font color="yellow">用于debug的输出残留</font>
+ <font color="yellow">测试阶段，联邦学习服务器将在连接两台客户机后自动开始分发模型并训练</font>

### <font>2023/3/29 更新</font>
+ 重构gan攻击模块并进行了单元测试: 对mnist数据集表现可行
  + 涉及文件: gan_attack/gan.py
  + (被gitignore)附加目录: gan_attack/images 用于保存攻击生成的图片
  + (被gitignore)附加目录: gan_attack/models 用于保存模型权重 (仅为节省测试时间)
+ gan模块等待整合至联邦学习中
+ <font color="yellow">由于gan攻击测试中发现生成图片有规则乱码区域，经检查后发现是原有cnn架构卷积核大小不合适导致边缘更新被略去，故将联邦学习中所用模型前两部卷积部分改为 5*5 卷积核。经测试，此改变未对项目表现造成影响</font>
    + 涉及文件 server.py

### <font>2023/4/11</font>
+ 将gan模块整合至联邦学习模型中，并进行了集成测试: 50次攻击结果可视化为1681206758.png
+ <font color="yellow">对原有联邦学习所使用的模型与聚合方式进行了修正以更好地体现攻击效果</font>
+ <font color="red">若想要在单一机器上调试，需改变cookies以改变当前身份，被标记为恶意的用户在进入联邦学习时将会自动进入攻击</font>

### <font color="yellow">2023/4/14</font>
+ 完成gan攻击及后续攻防对可视化部分所用接口，并完成集成测试
+ 删去原本static目录下gan产生的无用图片
+ <font color="red">当单机同时运行多个客户端时，由于客户端采用cookies用以保存用户，故在训练完成后会产生用户交叉的问题</font>

### <font color="red">2023/4/25</font>
+ 完成对梯度的差分隐私加噪，并进行了集成测试
+ 对差分隐私对正常训练的影响做评估，结果为static/images/evaluate_img/acc_xx.png系列图片
+ <font color="yellow">将余下的接口整合为标准化形态，但仍无用，建议只使用“差分隐私防御”与“GAN攻击”以作为实际运行时例子</font>