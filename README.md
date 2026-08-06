<img width="6134" height="1518" alt="数据流程图" src="https://github.com/user-attachments/assets/2ae17e9e-3eee-4391-92a0-3518e3a6d662" />
# FM-WideDeep-on-Criteo-and-MoiveLens
项目简介：
使用FM算法与WideDeep算法分别在Criteo与MoiveLens数据集上进行预测，得到基线模型.


数据集说明：
Criteo使用train.txt, MoiveLens使用ml-100k数据集.


模型流程图：
FM：
<img width="6134" height="1518" alt="数据流程图" src="https://github.com/user-attachments/assets/b2db7389-880f-48f8-8a76-d7ace3c07961" />

WideDeep：<img width="7278" height="1630" alt="模型内部数据流" src="https://github.com/user-attachments/assets/a448953a-95a4-4d08-a41c-28dba7808f0b" />


运行方式：
pip install -r requirements.txt + python FM.py + WideDeep on Criteo.py + WideDeep on MoiveLens.py


运行结果：
WideDeep on MoiveLens:
<img width="957" height="357" alt="运行结果" src="https://github.com/user-attachments/assets/59730d88-cd57-48c1-bbf7-7222e1c62f25" />



基于FM的CTR预测: [https://blog.csdn.net/Amarashi/article/details/163417299?spm=1011.2415.3001.5331]

基于WideDeep的CTR预测: [https://blog.csdn.net/Amarashi/article/details/163399394?spm=1011.2415.3001.5331]

基于FM的MoiveLens的评分预测: [https://blog.csdn.net/Amarashi/article/details/163483814?spm=1011.2415.3001.5331]

基于WideDeep的MoiveLens的评分预测:[https://blog.csdn.net/Amarashi/article/details/163542955spm=1011.2415.3001.5331]
