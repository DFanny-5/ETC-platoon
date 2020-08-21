Event-Trigger-Control (ETC) on vehicle platoon
===========

Required libraries
--------------
Tensorflow 1.X (please download the 1.X version to run the code)  
How to install tensorflow with pip in the terminal:   https://www.tensorflow.org/install/pip  

OpenAI Gym  
Instruction for downloading OpenAI Gym:    https://gym.openai.com/docs/  

Preparation  
1. The two seperate .py file `car_racing.py` and `car_dynamic.py` are used to repalce the original  `car_racing.py` and `car_dynamic.py` in `OpenAI/gym/envs/box2d` (The location is the place where the OpenAI Gym is downloaded to) 

2. Open the Terminal on your PC and choose to the file by using  
```
cd ddpg
```
```
./run_etc.sh
```
The two files car_racing.py and car_dynamic.py should be used to repalce the ar_racing.py and car_dynamic.py in the OpenAI/gym/envs/box2d.




#Then can use : ./run_etc.sh in the ddpg file to run the test.
