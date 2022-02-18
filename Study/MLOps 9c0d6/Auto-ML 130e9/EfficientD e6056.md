# EfficientDet

[google/automl](https://github.com/google/automl/blob/master/efficientdet/README.md)

## Object Detection Model

### **backbone 모델**

[10. transfer learning](https://nittaku.tistory.com/270)

- EfficientNets, VGGNet, YOLO, 모델같은 base 모델을 backbone 모델이라 부른다.

### **BiFPN**

[FPN (Feature Pyramid Network) 과 BiFPN](http://machinelearningkorea.com/2020/01/19/fpn-feature-pyramid-network-%EA%B3%BC-bifpn/)

- BiFPN은 FPN에서 레이어마다 가중치를 주어 좀더 각각의 층에 대한 해상도 정보가 잘 녹아낼수 있도록 하는 장치

※ FPN (Feature Pyramid Network)

![EfficientD%20e6056/Untitled.png](EfficientD%20e6056/Untitled.png)

![EfficientD%20e6056/Untitled%201.png](EfficientD%20e6056/Untitled%201.png)

각각의 단계에 컨볼루션 필터가 적용되는데, 희한하게도 반대방향으로 흐르는 부분이 있다.

→ 작은 해상도나 큰 해상도에서 얻을수 있는 특징을 적당히 섞으면 제대로 좀더 정교한 predict를 할수 있지 않을까? 하는 방법

그 외에 PANet, Fully-connected FPN, BiFPN 등이 있음.

![EfficientD%20e6056/Untitled%202.png](EfficientD%20e6056/Untitled%202.png)

![EfficientD%20e6056/Untitled%203.png](EfficientD%20e6056/Untitled%203.png)

- **BiFPN**

![EfficientD%20e6056/Untitled%204.png](EfficientD%20e6056/Untitled%204.png)

→ PANet 처럼 Bottom-Up, Top-Down을 동시에 지니며 모든 방향에 특성을 전달

**EfficientDet** 모델은 **Backbone**, **BiFPN**을 **Base**로 함.

![EfficientD%20e6056/Untitled%205.png](EfficientD%20e6056/Untitled%205.png)

- **Backbone**: we employ [EfficientNets](https://arxiv.org/abs/1905.11946) as our backbone networks.
- **BiFPN**: we propose BiFPN, a bi-directional feature network enhanced with fast normalization, which enables easy and fast feature fusion.
- **Scaling**: we use a single compound scaling factor to govern the depth, width, and resolution for all backbone, feature & prediction networks.

tflite 

[Tensorflow lite (TFlite)란?](https://m.blog.naver.com/PostView.nhn?blogId=rlawlwoong&logNo=222015935892&proxyReferer=https:%2F%2Fwww.google.com%2F)