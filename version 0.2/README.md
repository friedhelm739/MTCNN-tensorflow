# version 0.2更新文档


## 更新原因

1、采用矩阵算法加速预测后运算速度，更新后提高约为version0.1的6倍速度；
2、引入若干函数，增强MTCNN鲁棒性；
3、修复linux下路径报错bug、修复landmark镜像后关键点计算错误的bug、修复fc正则化的bug；


## 更新细节

1、修复Detector在Linux系统下路径报错的bug；
2、取消MTCNN_detector.detect_single_face 内显示单个网络预测时间；
3、MTCNN_detector.calibrate_box加入参量model_name，取消box限制条件；
4、Pnet每个scale中加入NMS，阈值0.5；
5、引入入pad函数，修复因box越界导致边界人脸漏识别问题，越界则直接调整到边界；
6、修改detect_P/R/Onet的空返回值为[]；
7、引入convert_to_square 函数，去重复人脸；
8、引入bounding_box函数，修改Detector滑窗法为矩阵法；
9、修改Detector预测landmark通道；
10、在MTCNN_detector初始化时初始化Detector；
11、修复关键点镜像计算错误的bug；
12、修复fc正则化loss图加载错误的bug；