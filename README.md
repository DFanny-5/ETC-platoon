环境安装
============
1.	tensorflow 1.x版本，必须是2以下版本，否则很多函数无法运行。  
教程参考，注意一定要是1.几的版本 https://www.tensorflow.org/install/pip  
2.	OpenAI Gym, 主要是box2d的环境必须下载，mujuco等下载失败也没事  
教程参考，https://gym.openai.com/docs/  
准备工作
===================
1.	克隆github仓库到本地：  
在控制台terminal中输入代码git clone https://github.com/DFanny-5/ETC-platoon.git  
2.	找到OpenAI Gym被下载到的根目录，找到box2d的文件路径（一般为gym/envs/box2d）。在box2d的目录下找到已有的car_racing.py 与car_dynamic.py 文件，将步骤1中下载的的car_racing.py 与car_dynamic.py 文件复制黏贴覆盖掉原box2d中的同名文件  
3.	打开控制台，进入下载的目录，输入cd ETC-platoon, cd ddpg进入ddpg文件夹，输入 ./run_etc.sh 开始训练

重要自设参数
===============
修改存储数据的txt file的名字、位置
-----------
打开 /ETC-platoon/ddpg/baselines/ddpg/training_ddpg.py ，第15行，txt_path后双引号内填写要实验结果的txt file的存储路径与存储名称  
修改小车个数/初始距离  
打开car_racing.py文件，第35行，number_agent 设置小车个数（因为第一辆车为固定匀速运动，所以number_agent = 2意味着只有一个agent）  

同样在car_racing.py文件，第38行，set_distane用于设置每两个小车之间的间隔，可以将间隔设置为一个随机数从而达到每次训练开始时的初始间隔距离不固定的效果  

如果设置在一次训练中，每次游戏的初始间隔距离，则在上一步基础上，在car_racing.py文件，第310行，initial_distance_apart也同时设置为随机数或固定值  
设置前车/后车初始速度
-----------
后车初始速度：  
car_dynamic.py文件，第95行，此处w.omega为设置除第一辆领跑的车以外的车辆的初始速度  

前车初始、持续速度：  
Car_dynamic.py文件，第200行，此处w.omega为设置第一辆领跑小车的速度，每一帧前车的轮胎转速都会强制刷新为此值  

设置跟车距离
----------------
Car_racing.py 文件中，第41行，desire_distance用于设置后车跟前车的距离保持多少时算作跟车成功（请注意，当两车距离小于7，则为后车追尾前车，所以想设置的跟车距离应加上7再赋值到desire_distance中）  

设置模型的保存周期与保存路径
Training_ddpg.py文件中，第137行用于设置每多少次训练完成后保存一次已训练模型，如代码中所用if count_fwj % 50000 == 0: 意味着每5万帧游戏完成后保存一次全部模型  

同样在此文件中，第138行，单引号内填写想要把模型文件保存到的路径和名称（请注意结尾一定要是.ckpt文件，此文件为tensorflow的特殊文件类型）  


