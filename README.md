# Introduction
We try some torchvision models and [AdasOptimizer](https://github.com/YanaiEliyahu/AdasOptimizer) with the SCUT-FBP5500 dataset.

# Usage
#### Install
```
pip install -r requirements.txt
```

#### Train
Firstly, download the dataset from [SCUT-FBP5500-Database-Release](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
Secondly, modify the config.py according to your needs.
Finally, run:
```
python train.py
```

#### Benchmark Evaluation
We set ResNet-18, ResNet-50, ResNeXt-50 as the benchmarks of the SCUT-FBP5500 dataset, and we evaluate the benchmark on various measurement metrics, including: Pearson correlation (PC), maximum absolute error (MAE), and root mean square error (RMSE). The evaluation results are shown in the following.

```
python benchmark.py
```

##### 5 Fold Cross Validation

|     PC     |      1     |      2     |      3     |      4     |      5     |   Average  |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNet-18  |   0.9184   |   0.9306   |   0.9024   | **0.9231** |   0.9034   |   0.9156   |
| ResNet-50  |   0.8866   |   0.9190   |   0.8743   |   0.9102   |   0.8743   |   0.8929   |
| ResNeXt-50 | **0.9289** | **0.9404** | **0.9030** |   0.9109   | **0.9182** | **0.9203** |

|    MAE     |      1     |      2     |      3     |      4     |      5     |   Average  |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNet-18  |   0.2066   |   0.2125   | **0.2494** | **0.1815** |   0.2263   |   0.2153   |
| ResNet-50  |   0.2508   |   0.2242   |   0.2643   |   0.2145   |   0.2715   |   0.2451   |
| ResNeXt-50 | **0.1889** | **0.1823** |   0.2542   |   0.2136   | **0.2138** | **0.2106** |

|    RMSE    |      1     |      2     |      3     |      4     |      5     |   Average  |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ResNet-18  |   0.2576   |   0.2658   | **0.3060** | **0.2411** |   0.2925   |   0.2726   |
| ResNet-50  |   0.3047   |   0.2774   |   0.3359   |   0.2604   |   0.3305   |   0.3018   |
| ResNeXt-50 | **0.2337** | **0.2338** |   0.3082   |   0.2594   | **0.2706** | **0.2611** |

##### 60% Training and 40% Testing

|      |   ResNet-18   | ResNet-50 |   ResNeXt-50   |
|:----:|:-------------:|:---------:|:--------------:|
|  PC  |  **0.8905**   |  0.8741   |     0.8865     |
|  MAE |    0.2571     |  0.2528   |   **0.2458**   |
| RMSE |    0.3514     |  0.3623   |   **0.3471**   |

#### Test
Create the `models` directory in the root project and download `shape_predictor_5_face_landmarks.dat.bz2` file

```sh
mkdir models
cd models
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
tar xjvf shape_predictor_5_face_landmarks.dat.bz2
```

Run test

```sh
usage: predict.py [-h] [-i IMAGE] [-m MODEL]

Facial beauty predictor.

optional arguments:
  -h, --help  show this help message and exit
  -i IMAGE    Image to be predicted.
  -m MODEL    The model path of facial beauty predictor.

cmd example:
python predict.py -i testPic/test.jpg -m models/resnet18_best_state.pt
```

#### References

https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

https://github.com/CharlesPikachu/isBeauty

https://github.com/YanaiEliyahu/AdasOptimizer

https://github.com/HikkaV/Precise-face-alignment