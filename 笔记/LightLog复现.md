# LightLog复现

## 环境

```sh
conda activate Log
```

配置如下

```
(log) ➜  Enhanced TCN for Log Anomaly Detection on the BGL Dataset git:(main) ✗ pip list
Package                      Version
---------------------------- -----------
absl-py                      2.1.0
aiohttp                      3.8.6
aiosignal                    1.3.1
astor                        0.8.1
astunparse                   1.6.3
async-timeout                4.0.3
asynctest                    0.13.0
attrs                        24.2.0
cachetools                   4.2.4
certifi                      2022.12.7
charset-normalizer           3.4.3
flatbuffers                  25.2.10
frozenlist                   1.3.3
gast                         0.2.2
gensim                       4.2.0
google-auth                  1.35.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.62.3
h5py                         2.10.0
idna                         3.10
importlib-metadata           6.7.0
joblib                       1.3.2
keras                        2.11.0
Keras-Applications           1.0.8
Keras-Preprocessing          1.1.2
libclang                     18.1.1
Markdown                     3.4.4
MarkupSafe                   2.1.5
multidict                    6.0.5
numpy                        1.18.5
oauthlib                     3.2.2
openai                       0.28.0
opt-einsum                   3.3.0
packaging                    24.0
pandas                       1.3.5
pip                          22.3.1
protobuf                     3.19.6
pyasn1                       0.5.1
pyasn1-modules               0.3.0
python-dateutil              2.9.0.post0
pytz                         2025.2
PyYAML                       6.0.1
requests                     2.31.0
requests-oauthlib            2.0.0
rsa                          4.9.1
sci_learn                    0.1.4
scikit-learn                 1.0.2
scipy                        1.4.1
setuptools                   65.6.3
six                          1.17.0
smart-open                   7.1.0
tensorboard                  2.0.2
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.0.0
tensorflow-estimator         2.0.1
tensorflow-io-gcs-filesystem 0.34.0
termcolor                    2.3.0
threadpoolctl                3.1.0
tqdm                         4.67.1
typing_extensions            4.7.1
urllib3                      2.0.7
Werkzeug                     2.2.3
wheel                        0.38.4
wrapt                        1.16.0
yarl                         1.9.4
zipp                         3.15.0
```



## 结构

```
➜  /workspace git:(main) ✗ ls
'BGL&HDFS dataset and Methods of data processing'
'Enhanced TCN for Log Anomaly Detection on the BGL Dataset'
'Enhanced TCN for Log Anomaly Detection on the HDFS Dataset'
 LICENSE
 pre_process.py
 README.md
 sequence.csv
 structured.csv
```

## 运行

### 训练

```
PCA降维度后的数据的维度： (378, 20)
2025-09-26 07:52:13.174899: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2025-09-26 07:52:13.178917: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2494140000 Hz
2025-09-26 07:52:13.179318: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2ec80280 executing computations on platform Host. Devices:
2025-09-26 07:52:13.179350: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 300, 20)]    0                                            
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 300, 3)       183         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 300, 1)       10          conv1d[0][0]                     
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 300, 3)       183         input_1[0][0]                    
__________________________________________________________________________________________________
add (Add)                       (None, 300, 3)       0           conv1d_1[0][0]                   
                                                                 conv1d_2[0][0]                   
__________________________________________________________________________________________________
activation (Activation)         (None, 300, 3)       0           add[0][0]                        
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 300, 3)       30          activation[0][0]                 
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 300, 1)       10          conv1d_3[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 300, 3)       0           conv1d_4[0][0]                   
                                                                 activation[0][0]                 
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 300, 3)       0           add_1[0][0]                      
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 300, 3)       30          activation_1[0][0]               
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 300, 1)       10          conv1d_5[0][0]                   
__________________________________________________________________________________________________
add_2 (Add)                     (None, 300, 3)       0           conv1d_6[0][0]                   
                                                                 activation_1[0][0]               
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 300, 3)       0           add_2[0][0]                      
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 300, 3)       30          activation_2[0][0]               
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 300, 1)       10          conv1d_7[0][0]                   
__________________________________________________________________________________________________
add_3 (Add)                     (None, 300, 3)       0           conv1d_8[0][0]                   
                                                                 activation_2[0][0]               
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 300, 3)       0           add_3[0][0]                      
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 3)            0           activation_3[0][0]               
__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            8           global_average_pooling1d[0][0]   
==================================================================================================
Total params: 504
Trainable params: 504
Non-trainable params: 0
__________________________________________________________________________________________________
WARNING:tensorflow:The `nb_epoch` argument in `fit` has been renamed `epochs`.
2025-09-26 07:52:13.468293: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 3349872000 exceeds 10% of system memory.
2025-09-26 07:52:15.878399: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 1435680000 exceeds 10% of system memory.
Train on 69789 samples, validate on 29910 samples
Epoch 1/100
69789/69789 - 30s - loss: 0.5742 - accuracy: 0.6714 - val_loss: 0.7470 - val_accuracy: 0.6358
Epoch 2/100
69789/69789 - 29s - loss: 0.5385 - accuracy: 0.6830 - val_loss: 0.7030 - val_accuracy: 0.6360
Epoch 3/100
69789/69789 - 29s - loss: 0.5307 - accuracy: 0.6869 - val_loss: 0.6846 - val_accuracy: 0.6278
Epoch 4/100
69789/69789 - 29s - loss: 0.5246 - accuracy: 0.6910 - val_loss: 0.7254 - val_accuracy: 0.6296
Epoch 5/100
69789/69789 - 29s - loss: 0.5198 - accuracy: 0.6924 - val_loss: 0.7502 - val_accuracy: 0.6298
Epoch 6/100
69789/69789 - 29s - loss: 0.5165 - accuracy: 0.6925 - val_loss: 0.7217 - val_accuracy: 0.6301
Epoch 7/100
69789/69789 - 29s - loss: 0.5117 - accuracy: 0.6926 - val_loss: 0.7222 - val_accuracy: 0.6301
Epoch 8/100
69789/69789 - 29s - loss: 0.5004 - accuracy: 0.6961 - val_loss: 0.7068 - val_accuracy: 0.6355
Epoch 9/100
69789/69789 - 29s - loss: 0.4889 - accuracy: 0.7107 - val_loss: 0.6398 - val_accuracy: 0.6542
Epoch 10/100
69789/69789 - 29s - loss: 0.4760 - accuracy: 0.7384 - val_loss: 0.6114 - val_accuracy: 0.6795
Epoch 11/100
69789/69789 - 29s - loss: 0.4747 - accuracy: 0.7452 - val_loss: 0.5974 - val_accuracy: 0.7110
Epoch 12/100
69789/69789 - 29s - loss: 0.4545 - accuracy: 0.7793 - val_loss: 0.5783 - val_accuracy: 0.7406
Epoch 13/100
69789/69789 - 29s - loss: 0.4480 - accuracy: 0.7858 - val_loss: 0.5705 - val_accuracy: 0.7508
Epoch 14/100
69789/69789 - 29s - loss: 0.4382 - accuracy: 0.7959 - val_loss: 0.5655 - val_accuracy: 0.7366
Epoch 15/100
69789/69789 - 29s - loss: 0.4219 - accuracy: 0.8122 - val_loss: 0.5465 - val_accuracy: 0.7599
Epoch 16/100
69789/69789 - 29s - loss: 0.4152 - accuracy: 0.8180 - val_loss: 0.5205 - val_accuracy: 0.7515
Epoch 17/100
69789/69789 - 29s - loss: 0.3995 - accuracy: 0.8330 - val_loss: 0.9791 - val_accuracy: 0.3972
Epoch 18/100
69789/69789 - 29s - loss: 0.3937 - accuracy: 0.8334 - val_loss: 0.5121 - val_accuracy: 0.8081
Epoch 19/100
69789/69789 - 29s - loss: 0.3827 - accuracy: 0.8431 - val_loss: 0.5120 - val_accuracy: 0.7929
Epoch 20/100
69789/69789 - 29s - loss: 0.3724 - accuracy: 0.8525 - val_loss: 0.5062 - val_accuracy: 0.8124
Epoch 21/100
69789/69789 - 29s - loss: 0.3781 - accuracy: 0.8450 - val_loss: 0.5221 - val_accuracy: 0.8086
Epoch 22/100
69789/69789 - 29s - loss: 0.3571 - accuracy: 0.8599 - val_loss: 0.5026 - val_accuracy: 0.8263
Epoch 23/100
69789/69789 - 29s - loss: 0.3569 - accuracy: 0.8551 - val_loss: 0.4920 - val_accuracy: 0.8206
Epoch 24/100
69789/69789 - 29s - loss: 0.3424 - accuracy: 0.8650 - val_loss: 0.5189 - val_accuracy: 0.8345
Epoch 25/100
69789/69789 - 29s - loss: 0.3605 - accuracy: 0.8537 - val_loss: 0.8639 - val_accuracy: 0.4179
Epoch 26/100
69789/69789 - 29s - loss: 0.3317 - accuracy: 0.8683 - val_loss: 0.4727 - val_accuracy: 0.8413
Epoch 27/100
69789/69789 - 29s - loss: 0.3776 - accuracy: 0.8391 - val_loss: 0.4618 - val_accuracy: 0.8262
Epoch 28/100
69789/69789 - 29s - loss: 0.3346 - accuracy: 0.8650 - val_loss: 0.4761 - val_accuracy: 0.8320
Epoch 29/100
69789/69789 - 29s - loss: 0.3411 - accuracy: 0.8646 - val_loss: 0.4753 - val_accuracy: 0.8305
Epoch 30/100
69789/69789 - 29s - loss: 0.3246 - accuracy: 0.8711 - val_loss: 0.4716 - val_accuracy: 0.8394
Epoch 31/100
69789/69789 - 29s - loss: 0.3252 - accuracy: 0.8712 - val_loss: 0.4521 - val_accuracy: 0.8377
Epoch 32/100
69789/69789 - 29s - loss: 0.3349 - accuracy: 0.8647 - val_loss: 0.4568 - val_accuracy: 0.8366
Epoch 33/100
69789/69789 - 29s - loss: 0.3484 - accuracy: 0.8588 - val_loss: 0.5249 - val_accuracy: 0.8084
Epoch 34/100
69789/69789 - 29s - loss: 0.3755 - accuracy: 0.8384 - val_loss: 0.6171 - val_accuracy: 0.7474
Epoch 35/100
69789/69789 - 29s - loss: 0.3950 - accuracy: 0.8349 - val_loss: 0.5813 - val_accuracy: 0.8152
Epoch 36/100
69789/69789 - 29s - loss: 0.3657 - accuracy: 0.8528 - val_loss: 0.4525 - val_accuracy: 0.8336
Epoch 37/100
69789/69789 - 29s - loss: 0.3403 - accuracy: 0.8653 - val_loss: 0.4577 - val_accuracy: 0.8423
Epoch 38/100
69789/69789 - 29s - loss: 0.3721 - accuracy: 0.8476 - val_loss: 0.5106 - val_accuracy: 0.8055
Epoch 39/100
69789/69789 - 29s - loss: 0.3328 - accuracy: 0.8616 - val_loss: 0.4915 - val_accuracy: 0.8342
Epoch 40/100
69789/69789 - 29s - loss: 0.3433 - accuracy: 0.8617 - val_loss: 0.4450 - val_accuracy: 0.8383
Epoch 41/100
69789/69789 - 29s - loss: 0.3135 - accuracy: 0.8759 - val_loss: 0.4728 - val_accuracy: 0.8473
Epoch 42/100
69789/69789 - 29s - loss: 0.3413 - accuracy: 0.8611 - val_loss: 0.5382 - val_accuracy: 0.7969
Epoch 43/100
69789/69789 - 29s - loss: 0.3637 - accuracy: 0.8556 - val_loss: 0.5194 - val_accuracy: 0.8134
Epoch 44/100
69789/69789 - 29s - loss: 0.3607 - accuracy: 0.8587 - val_loss: 0.4734 - val_accuracy: 0.8385
Epoch 45/100
69789/69789 - 29s - loss: 0.3374 - accuracy: 0.8637 - val_loss: 0.4670 - val_accuracy: 0.8296
Epoch 46/100
69789/69789 - 29s - loss: 0.3291 - accuracy: 0.8653 - val_loss: 0.4865 - val_accuracy: 0.8221
Epoch 47/100
69789/69789 - 29s - loss: 0.3021 - accuracy: 0.8799 - val_loss: 0.4270 - val_accuracy: 0.8432
Epoch 48/100
69789/69789 - 29s - loss: 0.3662 - accuracy: 0.8552 - val_loss: 0.5147 - val_accuracy: 0.8307
Epoch 49/100
69789/69789 - 29s - loss: 0.3808 - accuracy: 0.8445 - val_loss: 0.5434 - val_accuracy: 0.8199
Epoch 50/100
69789/69789 - 29s - loss: 0.3034 - accuracy: 0.8818 - val_loss: 0.4292 - val_accuracy: 0.8437
Epoch 51/100
69789/69789 - 29s - loss: 0.3211 - accuracy: 0.8730 - val_loss: 0.4253 - val_accuracy: 0.8421
Epoch 52/100
69789/69789 - 29s - loss: 0.4614 - accuracy: 0.7980 - val_loss: 0.6209 - val_accuracy: 0.7139
Epoch 53/100
69789/69789 - 29s - loss: 0.3841 - accuracy: 0.8287 - val_loss: 0.4703 - val_accuracy: 0.8426
Epoch 54/100
69789/69789 - 29s - loss: 0.3494 - accuracy: 0.8600 - val_loss: 0.4413 - val_accuracy: 0.8439
Epoch 55/100
69789/69789 - 29s - loss: 0.3036 - accuracy: 0.8803 - val_loss: 0.4549 - val_accuracy: 0.8381
Epoch 56/100
69789/69789 - 29s - loss: 0.2959 - accuracy: 0.8830 - val_loss: 0.4535 - val_accuracy: 0.8421
Epoch 57/100
69789/69789 - 29s - loss: 0.3915 - accuracy: 0.8278 - val_loss: 0.4510 - val_accuracy: 0.8413
Epoch 58/100
69789/69789 - 29s - loss: 0.2937 - accuracy: 0.8844 - val_loss: 0.4354 - val_accuracy: 0.8498
Epoch 59/100
69789/69789 - 29s - loss: 0.3337 - accuracy: 0.8704 - val_loss: 0.5491 - val_accuracy: 0.8076
Epoch 60/100
69789/69789 - 29s - loss: 0.3213 - accuracy: 0.8734 - val_loss: 0.4462 - val_accuracy: 0.8458
Epoch 61/100
69789/69789 - 29s - loss: 0.3008 - accuracy: 0.8824 - val_loss: 0.4917 - val_accuracy: 0.8359
Epoch 62/100
69789/69789 - 29s - loss: 0.3100 - accuracy: 0.8771 - val_loss: 0.4458 - val_accuracy: 0.8392
Epoch 63/100
69789/69789 - 29s - loss: 0.2889 - accuracy: 0.8848 - val_loss: 0.4342 - val_accuracy: 0.8450
Epoch 64/100
69789/69789 - 31s - loss: 0.3251 - accuracy: 0.8718 - val_loss: 0.5166 - val_accuracy: 0.8110
Epoch 65/100
69789/69789 - 45s - loss: 0.3359 - accuracy: 0.8632 - val_loss: 0.4493 - val_accuracy: 0.8427
Epoch 66/100
69789/69789 - 31s - loss: 0.3339 - accuracy: 0.8679 - val_loss: 0.5170 - val_accuracy: 0.8046
Epoch 67/100
69789/69789 - 30s - loss: 0.3446 - accuracy: 0.8602 - val_loss: 0.4685 - val_accuracy: 0.8225
Epoch 68/100
69789/69789 - 31s - loss: 0.3117 - accuracy: 0.8743 - val_loss: 0.4743 - val_accuracy: 0.8275
Epoch 69/100
69789/69789 - 30s - loss: 0.3572 - accuracy: 0.8586 - val_loss: 0.4620 - val_accuracy: 0.8144
Epoch 70/100
69789/69789 - 30s - loss: 0.3032 - accuracy: 0.8803 - val_loss: 0.4524 - val_accuracy: 0.8410
Epoch 71/100
69789/69789 - 30s - loss: 0.2794 - accuracy: 0.8907 - val_loss: 0.4692 - val_accuracy: 0.8453
Epoch 72/100
69789/69789 - 30s - loss: 0.3935 - accuracy: 0.8487 - val_loss: 0.4716 - val_accuracy: 0.8308
Epoch 73/100
69789/69789 - 29s - loss: 0.3247 - accuracy: 0.8704 - val_loss: 0.4786 - val_accuracy: 0.8194
Epoch 74/100
69789/69789 - 29s - loss: 0.3062 - accuracy: 0.8764 - val_loss: 0.4934 - val_accuracy: 0.8260
Epoch 75/100
69789/69789 - 29s - loss: 0.3631 - accuracy: 0.8560 - val_loss: 0.4896 - val_accuracy: 0.8350
Epoch 76/100
69789/69789 - 29s - loss: 0.2966 - accuracy: 0.8819 - val_loss: 0.5062 - val_accuracy: 0.8465
Epoch 77/100
69789/69789 - 29s - loss: 0.5368 - accuracy: 0.7422 - val_loss: 0.6496 - val_accuracy: 0.7029
Epoch 78/100
69789/69789 - 29s - loss: 0.4878 - accuracy: 0.7613 - val_loss: 0.6131 - val_accuracy: 0.7377
Epoch 79/100
69789/69789 - 29s - loss: 0.4655 - accuracy: 0.7813 - val_loss: 0.6087 - val_accuracy: 0.7299
Epoch 80/100
69789/69789 - 29s - loss: 0.4504 - accuracy: 0.7963 - val_loss: 0.6298 - val_accuracy: 0.7440
Epoch 81/100
69789/69789 - 29s - loss: 0.4304 - accuracy: 0.8092 - val_loss: 0.6486 - val_accuracy: 0.7508
Epoch 82/100
69789/69789 - 29s - loss: 0.4181 - accuracy: 0.8155 - val_loss: 0.6254 - val_accuracy: 0.7664
Epoch 83/100
69789/69789 - 29s - loss: 0.4093 - accuracy: 0.8193 - val_loss: 0.7243 - val_accuracy: 0.7542
Epoch 84/100
69789/69789 - 29s - loss: 0.4044 - accuracy: 0.8228 - val_loss: 0.6797 - val_accuracy: 0.7796
Epoch 85/100
69789/69789 - 29s - loss: 0.3983 - accuracy: 0.8245 - val_loss: 0.6703 - val_accuracy: 0.7719
Epoch 86/100
69789/69789 - 29s - loss: 0.3929 - accuracy: 0.8284 - val_loss: 0.6527 - val_accuracy: 0.7810
Epoch 87/100
69789/69789 - 29s - loss: 0.3880 - accuracy: 0.8298 - val_loss: 0.6282 - val_accuracy: 0.7691
Epoch 88/100
69789/69789 - 29s - loss: 0.3831 - accuracy: 0.8329 - val_loss: 0.6360 - val_accuracy: 0.7804
Epoch 89/100
69789/69789 - 29s - loss: 0.3827 - accuracy: 0.8328 - val_loss: 0.6151 - val_accuracy: 0.7865
Epoch 90/100
69789/69789 - 29s - loss: 0.3785 - accuracy: 0.8341 - val_loss: 0.5826 - val_accuracy: 0.7836
Epoch 91/100
69789/69789 - 29s - loss: 0.3771 - accuracy: 0.8357 - val_loss: 0.6121 - val_accuracy: 0.7852
Epoch 92/100
69789/69789 - 30s - loss: 0.3755 - accuracy: 0.8356 - val_loss: 0.5814 - val_accuracy: 0.7885
Epoch 93/100
69789/69789 - 29s - loss: 0.3720 - accuracy: 0.8391 - val_loss: 0.5676 - val_accuracy: 0.7944
Epoch 94/100
69789/69789 - 29s - loss: 0.3818 - accuracy: 0.8282 - val_loss: 0.5442 - val_accuracy: 0.7960
Epoch 95/100
69789/69789 - 29s - loss: 0.3691 - accuracy: 0.8401 - val_loss: 0.5708 - val_accuracy: 0.7975
Epoch 96/100
69789/69789 - 29s - loss: 0.3657 - accuracy: 0.8420 - val_loss: 0.5617 - val_accuracy: 0.7984
Epoch 97/100
69789/69789 - 29s - loss: 0.3642 - accuracy: 0.8416 - val_loss: 0.5404 - val_accuracy: 0.8055
Epoch 98/100
69789/69789 - 30s - loss: 0.3594 - accuracy: 0.8440 - val_loss: 0.5501 - val_accuracy: 0.8006
Epoch 99/100
69789/69789 - 30s - loss: 0.3614 - accuracy: 0.8432 - val_loss: 0.5285 - val_accuracy: 0.7935
Epoch 100/100
69789/69789 - 30s - loss: 0.3584 - accuracy: 0.8435 - val_loss: 0.5937 - val_accuracy: 0.8044
The model has been saved
```

