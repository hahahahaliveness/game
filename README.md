运行环境：
    linux
	cpu：4
	GPU：>=1
	memory: >=8
软件环境：
    python3
	torch 1.0.0
	tqdm
	glob
	torchvision 0.2.1
	cv2

将解压后的test数据集目录Testing 和 解压后的train数据集目录Training 和 解压后的validation数据集目录Val 软链到pytorch_model/config/pytorch.update.base.20190308/ 目录中

cd pytorch_model/config/pytorch.update.base.20190308/
train: python3 train.py (imagenet初始化)
val: python3 torch_inference_gpu.py val  
选出auc最高模型->bestModelPath, 执行以下命令排行第一的那个epoch的路径：python3 eval_val.py train_log/eval/*.json --threshold 1 --limit 20 
test：python3 torch_inference_gpu.py test bestModelPath

eval后的结果json转化为标准结果result.txt：
    python3 get_result.py train_log/eval/xx.json 0.5
