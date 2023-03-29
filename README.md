## 更新日志
### <font color="yellow">2023/3/19 更新</font>
+ 重构项目中以'server.py'与'client.py'为主体的联邦学习模块
+ 将联邦学习模块整合入服务器路由，现可通过'/fl_server'与'/fl_client'路由对服务器与客户端进行分别访问
+ 增加'logs'目录用以存放联邦学习产生的日志，联邦学习关闭后可自动删除对应日志 (计划之后各模块日志都附属于此目录下)
+ <font color="red">仍需讨论关于界面表现与联邦学习进入方式的问题</font>
+ <font color="red">无法关闭联邦学习服务器，现用守护进程保证其在主进程关闭时可释放资源，但无法即时关闭此服务器</font>
+ <font color="red">无法将联邦学习过程中产生的tensorflow日志重定位至标准日志文件，会导致前端无法清晰地显示训练过程</font>
+ <font color="yellow">用于debug的输出残留</font>
+ <font color="yellow">测试阶段，联邦学习服务器将在连接两台客户机后自动开始分发模型并训练</font>

### <font color="red">2023/3/29 更新</font>
+ 重构gan攻击模块并进行了单元测试: 对mnist数据集表现可行
  + 涉及文件: gan_attack/gan.py
  + (被gitignore)附加目录: gan_attack/images 用于保存攻击生成的图片
  + (被gitignore)附加目录: gan_attack/models 用于保存模型权重 (仅为节省测试时间)
+ gan模块等待整合至联邦学习中
+ <font color="yellow">由于gan攻击测试中发现生成图片有规则乱码区域，经检查后发现是原有cnn架构卷积核大小不合适导致边缘更新被略去，故将联邦学习中所用模型前两部卷积部分改为 5*5 卷积核。经测试，此改变未对项目表现造成影响</font>
    + 涉及文件 server.py
