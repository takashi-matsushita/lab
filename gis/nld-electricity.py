import os

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
plt.style.use('bmh') 

import geoviews as gv
gv.extension('bokeh')


province = {
'Drenthe':          '7705 7740 7741 7742 7750 7751 7753 7754 7755 7756 7760 7761 7764 7765 7766 7800 7801 7811 7812 7813 7814 7815 7821 7822 7823 7824 7825 7826 7827 7828 7830 7831 7833 7840 7841 7842 7843 7844 7845 7846 7847 7848 7849 7851 7852 7853 7854 7855 7856 7858 7859 7860 7861 7863 7864 7871 7872 7873 7874 7875 7876 7877 7880 7881 7884 7885 7887 7889 7890 7891 7892 7894 7895 7900 7901 7902 7903 7904 7905 7906 7907 7908 7909 7910 7911 7912 7913 7914 7915 7916 7917 7918 7920 7921 7924 7925 7926 7927 7928 7929 7931 7932 7933 7934 7935 7936 7937 7938 7940 7941 7942 7943 7944 7948 7949 7957 7958 7960 7961 7963 7964 7965 7966 7970 7971 7973 7974 7975 7980 7981 7983 7984 7985 7986 7990 7991 8351 8380 8381 8382 8383 8384 8385 8386 8387 8420 8421 8422 8423 8424 8425 8426 8427 8428 8430 8431 8432 8433 8434 8435 8437 8438 8439 9300 9301 9302 9304 9305 9306 9307 9311 9312 9313 9314 9315 9320 9321 9330 9331 9333 9334 9335 9336 9337 9341 9342 9343 9351 9400 9401 9402 9403 9404 9405 9406 9407 9408 9409 9410 9411 9412 9413 9414 9415 9416 9417 9418 9419 9420 9421 9422 9423 9430 9431 9432 9433 9434 9435 9436 9437 9438 9439 9441 9442 9443 9444 9445 9446 9447 9448 9449 9450 9451 9452 9453 9454 9455 9456 9457 9458 9459 9460 9461 9462 9463 9464 9465 9466 9467 9468 9469 9470 9471 9472 9473 9474 9475 9480 9481 9482 9483 9484 9485 9486 9487 9488 9489 9491 9492 9493 9494 9495 9496 9497 9511 9512 9514 9515 9520 9521 9523 9524 9525 9526 9527 9528 9530 9531 9533 9534 9535 9536 9537 9564 9571 9573 9574 9654 9655 9656 9657 9658 9659 9749 9760 9761 9765 9766 9780 9781 9784 9785 9959',
'Flevoland':        '1300 1301 1302 1303 1305 1309 1311 1312 1313 1314 1315 1316 1317 1318 1319 1320 1321 1322 1323 1324 1325 1326 1327 1328 1329 1331 1332 1333 1334 1335 1336 1338 1339 1341 1343 1349 1351 1352 1353 1354 1355 1356 1357 1358 1359 1361 1362 1363 3890 3891 3892 3893 3894 3895 3896 3897 3898 3899 8200 8202 8203 8211 8212 8218 8219 8221 8222 8223 8224 8225 8226 8231 8232 8233 8239 8241 8242 8243 8244 8245 8250 8251 8252 8253 8254 8255 8256 8300 8301 8302 8303 8304 8305 8307 8308 8309 8311 8312 8313 8314 8315 8316 8317 8319 8320 8321 8322',
'Friesland':        '1794 3925 8388 8389 8390 8391 8392 8393 8394 8395 8396 8397 8398 8400 8401 8403 8404 8405 8406 8407 8408 8409 8410 8411 8412 8413 8414 8415 8431 8440 8441 8442 8443 8444 8445 8446 8447 8448 8449 8451 8452 8453 8454 8455 8456 8457 8458 8459 8461 8462 8463 8464 8465 8466 8467 8468 8469 8470 8471 8472 8474 8475 8476 8477 8478 8479 8481 8482 8483 8484 8485 8486 8487 8488 8489 8490 8491 8493 8494 8495 8497 8500 8501 8502 8503 8505 8506 8507 8508 8511 8512 8513 8514 8515 8516 8517 8520 8521 8522 8523 8524 8525 8526 8527 8528 8529 8530 8531 8532 8534 8535 8536 8537 8538 8539 8541 8542 8550 8551 8552 8553 8554 8556 8560 8561 8563 8564 8565 8566 8567 8571 8572 8573 8574 8576 8581 8582 8583 8584 8600 8601 8602 8603 8604 8605 8606 8607 8608 8611 8612 8613 8614 8615 8616 8617 8618 8620 8621 8622 8623 8624 8625 8626 8627 8628 8629 8631 8632 8633 8644 8647 8650 8651 8658 8700 8701 8702 8710 8711 8713 8715 8721 8722 8723 8724 8741 8742 8743 8744 8745 8746 8747 8748 8749 8751 8752 8753 8754 8755 8756 8757 8758 8759 8761 8762 8763 8764 8765 8766 8771 8772 8773 8774 8775 8782 8800 8801 8802 8804 8805 8806 8807 8808 8809 8811 8812 8813 8814 8816 8821 8822 8823 8850 8851 8852 8853 8854 8855 8856 8857 8860 8861 8862 8871 8872 8880 8881 8882 8883 8884 8885 8890 8891 8892 8893 8894 8895 8896 8897 8899 8900 8901 8902 8903 8911 8912 8913 8914 8915 8916 8917 8918 8919 8921 8922 8923 8924 8925 8926 8927 8931 8932 8933 8934 8935 8936 8937 8938 8939 8941 9000 9001 9003 9004 9005 9006 9007 9008 9009 9011 9012 9013 9014 9031 9032 9033 9034 9035 9036 9037 9038 9040 9041 9043 9044 9045 9047 9050 9051 9053 9054 9055 9056 9057 9060 9061 9062 9063 9064 9067 9071 9072 9073 9074 9075 9076 9077 9078 9079 9081 9082 9083 9084 9085 9086 9087 9088 9089 9091 9100 9101 9102 9103 9104 9105 9106 9107 9108 9109 9111 9112 9113 9114 9121 9122 9123 9124 9125 9131 9132 9133 9134 9135 9136 9137 9138 9141 9142 9143 9144 9145 9146 9147 9148 9150 9151 9152 9153 9154 9155 9156 9160 9161 9162 9163 9164 9166 9171 9172 9173 9174 9175 9176 9177 9178 9200 9201 9202 9203 9204 9205 9206 9207 9211 9212 9213 9214 9215 9216 9217 9218 9219 9221 9222 9223 9230 9231 9233 9240 9241 9243 9244 9245 9246 9247 9248 9249 9250 9251 9254 9255 9256 9257 9258 9260 9261 9262 9263 9264 9265 9269 9270 9271 9280 9281 9283 9284 9285 9286 9287 9288 9289 9290 9291 9292 9293 9294 9295 9296 9297 9298 9299 9851 9852 9853 9871 9872 9873',
'Gelderland':       '1250 3770 3771 3772 3773 3774 3775 3776 3780 3781 3784 3785 3790 3792 3794 3840 3841 3842 3843 3844 3845 3846 3847 3848 3849 3850 3851 3852 3853 3860 3861 3862 3863 3864 3870 3871 3880 3881 3882 3886 3888 3920 3925 3931 4000 4001 4002 4003 4004 4005 4006 4007 4010 4011 4012 4013 4014 4016 4017 4020 4021 4023 4024 4030 4031 4032 4033 4040 4041 4043 4050 4051 4053 4054 4060 4061 4062 4063 4064 4100 4101 4102 4103 4104 4105 4106 4107 4110 4111 4112 4115 4116 4117 4119 4147 4151 4152 4153 4155 4156 4157 4158 4161 4170 4171 4174 4175 4176 4180 4181 4182 4184 4185 4190 4191 4194 4196 4197 4211 4212 4214 5300 5301 5302 5305 5306 5307 5308 5310 5311 5313 5314 5315 5316 5317 5318 5320 5321 5324 5325 5327 5328 5330 5331 5333 5334 5335 5855 6500 6501 6503 6504 6511 6512 6515 6521 6522 6523 6524 6525 6531 6532 6533 6534 6535 6536 6537 6538 6541 6542 6543 6544 6545 6546 6550 6551 6561 6562 6564 6566 6571 6572 6573 6574 6575 6576 6577 6578 6579 6580 6581 6582 6600 6601 6602 6603 6604 6605 6606 6610 6611 6612 6613 6615 6616 6617 6620 6621 6624 6626 6627 6628 6629 6631 6634 6640 6641 6642 6644 6645 6650 6651 6652 6653 6654 6655 6657 6658 6659 6660 6661 6662 6663 6665 6666 6668 6669 6670 6671 6672 6673 6674 6675 6676 6677 6678 6680 6681 6684 6685 6686 6687 6690 6691 6700 6701 6702 6703 6704 6705 6706 6707 6708 6709 6710 6711 6712 6713 6714 6715 6716 6717 6718 6720 6721 6730 6731 6732 6733 6740 6741 6744 6745 6800 6801 6802 6803 6811 6812 6813 6814 6815 6816 6821 6822 6823 6824 6825 6826 6827 6828 6831 6832 6833 6834 6835 6836 6841 6842 6843 6844 6845 6846 6850 6851 6852 6860 6861 6862 6865 6866 6869 6870 6871 6874 6877 6880 6881 6882 6883 6891 6900 6901 6902 6903 6904 6905 6909 6911 6913 6914 6915 6916 6917 6920 6921 6922 6923 6924 6930 6931 6932 6940 6941 6942 6950 6951 6952 6953 6955 6956 6957 6960 6961 6964 6970 6971 6974 6975 6980 6981 6982 6983 6984 6986 6987 6988 6990 6991 6994 6996 6997 6998 6999 7000 7001 7002 7003 7004 7005 7006 7007 7008 7009 7010 7011 7020 7021 7025 7030 7031 7035 7036 7037 7038 7039 7040 7041 7044 7045 7046 7047 7048 7050 7051 7054 7055 7060 7061 7064 7065 7070 7071 7075 7076 7077 7078 7080 7081 7083 7084 7090 7091 7095 7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7113 7115 7119 7120 7121 7122 7123 7126 7130 7131 7132 7134 7135 7136 7137 7140 7141 7142 7150 7151 7152 7156 7157 7160 7161 7165 7200 7201 7202 7203 7204 7205 7206 7207 7210 7211 7213 7214 7215 7216 7217 7218 7220 7221 7223 7224 7225 7226 7227 7230 7231 7232 7233 7234 7240 7241 7242 7244 7245 7250 7251 7255 7256 7260 7261 7263 7270 7271 7273 7274 7275 7300 7301 7302 7303 7311 7312 7313 7314 7315 7316 7317 7320 7321 7322 7323 7324 7325 7326 7327 7328 7329 7331 7332 7333 7334 7335 7336 7339 7341 7345 7346 7348 7350 7351 7352 7360 7361 7364 7370 7371 7380 7381 7382 7383 7384 7390 7391 7392 7395 7396 7397 7399 7439 7574 7580 7581 7582 7585 7586 7587 7588 8050 8051 8052 8070 8071 8072 8075 8076 8077 8079 8080 8081 8082 8084 8085 8090 8091 8094 8095 8096 8097 8160 8161 8162 8166 8167 8170 8171 8172 8180 8181 8190 8191 8193 8194 8461',
'Groningen':        '2750 2751 2752 2760 2761 2811 2840 2841 2910 2911 2912 2913 2914 5340 5341 5342 5343 5344 5345 5346 5347 5348 5349 5350 5351 5352 5353 5354 5355 5356 5357 5358 5359 5366 5367 5368 5370 5371 5373 5386 5394 5395 5396 5397 5398 9350 9351 9354 9355 9356 9359 9361 9362 9363 9364 9365 9366 9367 9479 9500 9501 9502 9503 9540 9541 9545 9550 9551 9560 9561 9563 9566 9580 9581 9584 9585 9591 9600 9601 9602 9603 9605 9606 9607 9608 9609 9610 9611 9613 9614 9615 9616 9617 9618 9619 9620 9621 9622 9623 9624 9625 9626 9627 9628 9629 9631 9632 9633 9635 9636 9640 9641 9642 9644 9645 9646 9648 9649 9651 9661 9663 9665 9670 9671 9672 9673 9674 9675 9677 9678 9679 9681 9682 9684 9685 9686 9687 9688 9691 9693 9695 9696 9697 9698 9699 9700 9701 9702 9703 9704 9711 9712 9713 9714 9715 9716 9717 9718 9721 9722 9723 9724 9725 9726 9727 9728 9731 9732 9733 9734 9735 9736 9737 9738 9741 9742 9743 9744 9745 9746 9747 9750 9751 9752 9753 9755 9756 9771 9773 9774 9790 9791 9792 9793 9794 9795 9796 9797 9798 9800 9801 9804 9805 9811 9812 9821 9822 9824 9825 9827 9828 9831 9832 9833 9841 9842 9843 9844 9845 9860 9861 9862 9863 9864 9865 9866 9881 9882 9883 9884 9885 9886 9891 9892 9893 9900 9901 9902 9903 9904 9905 9906 9907 9908 9909 9911 9912 9913 9914 9915 9917 9918 9919 9921 9922 9923 9924 9925 9930 9931 9932 9933 9934 9936 9937 9939 9942 9943 9944 9945 9946 9947 9948 9949 9951 9953 9954 9955 9956 9957 9961 9962 9963 9964 9965 9966 9967 9968 9969 9970 9971 9972 9973 9974 9975 9976 9977 9978 9979 9980 9981 9982 9983 9984 9985 9986 9987 9988 9989 9990 9991 9992 9993 9994 9995 9996 9997 9998 9999',
'Limburg':          '3197 5591 5766 5768 5800 5801 5802 5803 5804 5807 5808 5809 5811 5812 5813 5814 5815 5816 5817 5851 5853 5854 5855 5856 5860 5861 5862 5863 5864 5865 5866 5871 5872 5900 5901 5902 5911 5912 5913 5914 5915 5916 5921 5922 5923 5924 5925 5926 5927 5928 5930 5931 5932 5935 5940 5941 5943 5944 5950 5951 5953 5954 5960 5961 5962 5963 5964 5966 5970 5971 5973 5975 5976 5977 5980 5981 5984 5985 5986 5987 5988 5990 5991 5993 5995 6000 6001 6002 6003 6004 6005 6006 6011 6012 6013 6014 6015 6017 6019 6030 6031 6034 6035 6037 6039 6040 6041 6042 6043 6044 6045 6049 6050 6051 6060 6061 6063 6065 6067 6070 6071 6074 6075 6077 6080 6081 6082 6083 6085 6086 6088 6089 6091 6092 6093 6095 6096 6097 6099 6100 6101 6102 6104 6105 6107 6109 6111 6112 6114 6116 6118 6120 6121 6122 6123 6124 6125 6127 6129 6130 6131 6132 6133 6134 6135 6136 6137 6141 6142 6143 6151 6153 6155 6160 6161 6162 6163 6164 6165 6166 6167 6170 6171 6174 6176 6180 6181 6190 6191 6199 6200 6201 6202 6203 6211 6212 6213 6214 6215 6216 6217 6218 6219 6221 6222 6223 6224 6225 6226 6227 6228 6229 6230 6231 6235 6237 6240 6241 6243 6245 6247 6251 6252 6255 6261 6262 6265 6267 6268 6269 6270 6271 6273 6274 6276 6277 6278 6281 6285 6286 6287 6289 6290 6291 6294 6295 6301 6305 6307 6311 6312 6320 6321 6325 6333 6336 6341 6342 6343 6350 6351 6353 6360 6361 6363 6365 6367 6369 6370 6371 6372 6373 6374 6400 6401 6411 6412 6413 6414 6415 6416 6417 6418 6419 6421 6422 6430 6431 6432 6433 6436 6438 6439 6440 6441 6442 6443 6444 6445 6446 6447 6450 6451 6454 6456 6460 6461 6462 6463 6464 6465 6466 6467 6468 6469 6470 6471 6584 6585 6586 6587 6590 6591 6595 6596 6598 6599 7037',
'Noord-Brabant':    '4250 4251 4254 4255 4260 4261 4264 4265 4266 4267 4268 4269 4270 4271 4273 4280 4281 4283 4284 4285 4286 4287 4288 4600 4601 4602 4611 4612 4613 4614 4615 4616 4617 4621 4622 4623 4624 4625 4630 4631 4634 4635 4640 4641 4645 4650 4651 4652 4655 4660 4661 4664 4670 4671 4681 4700 4701 4702 4703 4704 4705 4706 4707 4708 4709 4710 4711 4714 4715 4721 4722 4724 4725 4726 4727 4730 4731 4735 4740 4741 4744 4750 4751 4754 4756 4758 4759 4760 4761 4762 4765 4766 4771 4772 4780 4781 4782 4790 4791 4793 4794 4796 4797 4800 4801 4802 4803 4811 4812 4813 4814 4815 4816 4817 4818 4819 4820 4822 4823 4824 4825 4826 4827 4834 4835 4836 4837 4838 4839 4840 4841 4844 4845 4847 4849 4850 4851 4854 4855 4856 4858 4859 4860 4861 4870 4871 4872 4873 4874 4875 4876 4877 4878 4879 4880 4881 4882 4884 4885 4890 4891 4900 4901 4902 4903 4904 4905 4906 4907 4908 4909 4911 4920 4921 4924 4926 4927 4930 4931 4940 4941 4942 4944 5000 5001 5002 5003 5004 5011 5012 5013 5014 5015 5017 5018 5021 5022 5025 5026 5032 5035 5036 5037 5038 5041 5042 5043 5044 5045 5046 5047 5048 5049 5050 5051 5052 5053 5056 5057 5059 5060 5061 5062 5063 5066 5070 5071 5074 5076 5080 5081 5084 5085 5087 5089 5090 5091 5094 5095 5096 5100 5101 5102 5103 5104 5105 5106 5107 5109 5110 5111 5113 5114 5120 5121 5122 5124 5125 5126 5130 5131 5133 5140 5141 5142 5143 5144 5145 5146 5150 5151 5152 5154 5160 5161 5162 5165 5170 5171 5172 5175 5176 5200 5201 5202 5203 5211 5212 5213 5215 5216 5221 5222 5223 5224 5231 5232 5233 5234 5235 5236 5237 5240 5241 5242 5243 5244 5245 5246 5247 5248 5249 5250 5251 5252 5253 5254 5256 5258 5260 5261 5262 5263 5264 5266 5268 5270 5271 5272 5275 5280 5281 5282 5283 5290 5291 5292 5293 5294 5296 5298 5360 5361 5363 5364 5374 5375 5381 5382 5383 5384 5386 5388 5390 5391 5392 5400 5401 5402 5403 5404 5405 5406 5408 5409 5410 5411 5420 5421 5422 5423 5424 5425 5427 5428 5430 5431 5432 5433 5434 5435 5437 5438 5439 5441 5443 5445 5446 5447 5449 5450 5451 5453 5454 5455 5461 5462 5463 5464 5465 5466 5467 5469 5471 5472 5473 5476 5481 5482 5491 5492 5500 5501 5502 5503 5504 5505 5506 5507 5508 5509 5511 5512 5513 5520 5521 5524 5525 5527 5528 5529 5530 5531 5534 5540 5541 5550 5551 5552 5553 5554 5555 5556 5560 5561 5563 5570 5571 5575 5580 5581 5582 5583 5590 5591 5595 5600 5601 5602 5603 5604 5605 5606 5611 5612 5613 5614 5615 5616 5617 5621 5622 5623 5624 5625 5626 5627 5628 5629 5631 5632 5633 5641 5642 5643 5644 5645 5646 5647 5651 5652 5653 5654 5655 5656 5657 5658 5660 5661 5662 5663 5664 5665 5666 5667 5670 5671 5672 5673 5674 5680 5681 5682 5683 5684 5685 5688 5689 5690 5691 5692 5694 5700 5701 5702 5703 5704 5705 5706 5707 5708 5709 5710 5711 5712 5715 5720 5721 5724 5725 5730 5731 5735 5737 5738 5740 5741 5750 5751 5752 5753 5754 5756 5757 5758 5759 5760 5761 5763 5764 5820 5821 5823 5824 5825 5826 5827 5830 5831 5835 5836 5840 5841 5843 5844 5845 5846 6020 6021 6023 6024 6026 6027 6028 6029 6626 6678',
'Noord-Holland':    '1000 1001 1002 1003 1005 1006 1007 1008 1009 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1030 1031 1032 1033 1034 1035 1036 1037 1040 1041 1042 1043 1044 1045 1046 1047 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1086 1087 1090 1091 1092 1093 1094 1095 1096 1097 1098 1100 1101 1102 1103 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1117 1118 1119 1120 1121 1127 1130 1131 1132 1135 1140 1141 1145 1150 1151 1153 1154 1156 1160 1161 1165 1170 1171 1175 1180 1181 1182 1183 1184 1185 1186 1187 1188 1189 1190 1191 1200 1201 1202 1211 1212 1213 1214 1215 1216 1217 1218 1221 1222 1223 1230 1231 1240 1241 1243 1244 1251 1252 1260 1261 1262 1270 1271 1272 1273 1274 1275 1276 1277 1380 1381 1382 1383 1384 1394 1398 1399 1400 1401 1402 1403 1404 1405 1406 1410 1411 1412 1420 1421 1422 1423 1424 1430 1431 1432 1433 1435 1436 1437 1438 1440 1441 1442 1443 1444 1445 1446 1447 1448 1451 1452 1454 1456 1458 1461 1462 1463 1464 1471 1472 1473 1474 1475 1476 1477 1481 1482 1483 1484 1485 1486 1487 1488 1489 1500 1501 1502 1503 1504 1505 1506 1507 1508 1509 1510 1511 1520 1521 1525 1530 1531 1534 1536 1540 1541 1544 1546 1550 1551 1560 1561 1562 1566 1567 1600 1601 1602 1606 1607 1608 1609 1610 1611 1613 1614 1616 1617 1619 1620 1621 1622 1623 1624 1625 1627 1628 1631 1633 1634 1636 1641 1642 1643 1645 1646 1647 1648 1652 1654 1655 1657 1658 1661 1662 1663 1670 1671 1674 1676 1678 1679 1681 1682 1683 1684 1685 1686 1687 1688 1689 1691 1692 1693 1695 1696 1697 1700 1701 1702 1703 1704 1705 1706 1710 1711 1713 1715 1716 1718 1719 1720 1721 1722 1723 1724 1730 1731 1732 1733 1734 1735 1736 1738 1740 1741 1742 1744 1746 1747 1749 1750 1751 1752 1753 1754 1755 1756 1757 1759 1760 1761 1764 1766 1767 1768 1769 1770 1771 1773 1774 1775 1777 1778 1779 1780 1781 1782 1783 1784 1785 1786 1787 1788 1789 1790 1791 1792 1793 1794 1795 1796 1797 1800 1801 1802 1810 1811 1812 1813 1814 1815 1816 1817 1821 1822 1823 1824 1825 1826 1827 1829 1830 1831 1832 1834 1840 1841 1842 1843 1844 1846 1847 1850 1851 1852 1853 1860 1861 1862 1865 1870 1871 1873 1900 1901 1902 1906 1910 1911 1920 1921 1930 1931 1934 1935 1940 1941 1942 1943 1944 1945 1946 1947 1948 1949 1950 1951 1960 1961 1962 1963 1964 1965 1966 1967 1968 1969 1970 1971 1972 1973 1974 1975 1976 1980 1981 1985 1990 1991 1992 2000 2001 2002 2003 2011 2012 2013 2014 2015 2019 2021 2022 2023 2024 2025 2026 2031 2032 2033 2034 2035 2036 2037 2040 2041 2042 2050 2051 2060 2061 2063 2064 2065 2070 2071 2080 2082 2100 2101 2102 2103 2104 2105 2106 2110 2111 2114 2116 2120 2121 2130 2131 2132 2133 2134 2135 2136 2140 2141 2142 2143 2144 2150 2151 2152 2153 2154 2155 2156 2157 2158 2165 3180 3625 3744 8896',
'Overijssel':       '5328 7255 7400 7401 7411 7412 7413 7414 7415 7416 7417 7418 7419 7420 7421 7422 7423 7424 7425 7426 7427 7428 7429 7430 7431 7433 7434 7435 7437 7440 7441 7442 7443 7447 7448 7450 7451 7460 7461 7462 7463 7466 7467 7468 7470 7471 7472 7475 7478 7480 7481 7482 7483 7490 7491 7495 7496 7497 7500 7502 7503 7504 7511 7512 7513 7514 7521 7522 7523 7524 7525 7531 7532 7533 7534 7535 7536 7541 7542 7543 7544 7545 7546 7547 7548 7550 7551 7552 7553 7554 7555 7556 7557 7558 7559 7561 7562 7570 7571 7572 7573 7574 7575 7576 7577 7590 7591 7595 7596 7597 7600 7601 7602 7603 7604 7605 7606 7607 7608 7609 7610 7611 7614 7615 7620 7621 7622 7623 7625 7626 7627 7630 7631 7634 7635 7636 7637 7638 7640 7641 7642 7645 7650 7651 7661 7662 7663 7664 7665 7666 7667 7668 7670 7671 7672 7675 7676 7678 7679 7680 7681 7683 7685 7686 7687 7688 7690 7691 7692 7693 7694 7695 7696 7700 7701 7702 7707 7710 7711 7715 7720 7721 7722 7730 7731 7732 7734 7735 7736 7737 7738 7739 7770 7771 7772 7773 7775 7776 7777 7778 7779 7780 7781 7782 7783 7784 7785 7786 7787 7788 7791 7792 7793 7794 7795 7796 7797 7798 7946 7950 7951 7954 7955 8000 8001 8002 8003 8004 8007 8011 8012 8013 8014 8015 8016 8017 8019 8021 8022 8023 8024 8025 8026 8028 8031 8032 8033 8034 8035 8041 8042 8043 8044 8045 8055 8060 8061 8064 8066 8100 8101 8102 8103 8105 8106 8107 8110 8111 8112 8120 8121 8124 8130 8131 8140 8141 8144 8146 8147 8148 8150 8151 8152 8153 8154 8196 8198 8260 8261 8262 8263 8264 8265 8266 8267 8271 8274 8275 8276 8277 8278 8280 8281 8291 8293 8294 8325 8326 8330 8331 8332 8333 8334 8335 8336 8337 8338 8339 8341 8342 8343 8344 8345 8346 8347 8355 8356 8361 8362 8363 8371 8372 8373 8374 8375 8376 8377 8378',
'Utrecht':          '1390 1391 1393 1396 1426 1427 3400 3401 3402 3403 3404 3405 3410 3411 3412 3413 3415 3417 3420 3421 3425 3430 3431 3432 3433 3434 3435 3436 3437 3438 3439 3440 3441 3442 3443 3444 3445 3446 3447 3448 3449 3450 3451 3452 3453 3454 3455 3460 3461 3464 3467 3470 3471 3474 3480 3481 3500 3501 3502 3503 3504 3505 3506 3507 3508 3509 3511 3512 3513 3514 3515 3521 3522 3523 3524 3525 3526 3527 3528 3531 3532 3533 3534 3540 3541 3542 3543 3544 3545 3546 3551 3552 3553 3554 3555 3561 3562 3563 3564 3565 3566 3571 3572 3573 3581 3582 3583 3584 3585 3600 3601 3602 3603 3604 3605 3606 3607 3608 3611 3612 3615 3620 3621 3626 3628 3630 3631 3632 3633 3634 3640 3641 3642 3643 3645 3646 3648 3700 3701 3702 3703 3704 3705 3706 3707 3708 3709 3710 3711 3712 3720 3721 3722 3723 3730 3731 3732 3734 3735 3737 3738 3739 3740 3741 3742 3743 3744 3749 3750 3751 3752 3754 3755 3760 3761 3762 3763 3764 3765 3766 3768 3769 3791 3800 3802 3811 3812 3813 3814 3815 3816 3817 3818 3819 3821 3822 3823 3824 3825 3826 3828 3829 3830 3831 3832 3833 3834 3835 3836 3900 3901 3902 3903 3904 3905 3906 3907 3910 3911 3912 3921 3922 3927 3930 3931 3940 3941 3945 3947 3950 3951 3953 3956 3958 3959 3960 3961 3962 3970 3971 3972 3980 3981 3984 3985 3989 3990 3991 3992 3993 3994 3995 3997 3998 3999 4121 4122 4124 4125 4130 4131 4132 4133',
'Zeeland':          '4300 4301 4302 4303 4305 4306 4307 4308 4310 4311 4315 4316 4317 4318 4321 4322 4323 4325 4326 4327 4328 4330 4331 4332 4333 4334 4335 4336 4337 4338 4339 4340 4341 4350 4351 4352 4353 4354 4356 4357 4360 4361 4363 4364 4365 4370 4371 4373 4374 4380 4381 4382 4383 4384 4385 4386 4387 4388 4389 4400 4401 4410 4411 4413 4414 4415 4416 4417 4420 4421 4423 4424 4430 4431 4433 4434 4435 4436 4437 4438 4440 4441 4443 4444 4450 4451 4453 4454 4455 4456 4458 4460 4461 4462 4463 4464 4465 4470 4471 4472 4474 4475 4481 4482 4484 4485 4486 4490 4491 4493 4494 4500 4501 4503 4504 4505 4506 4507 4508 4510 4511 4513 4515 4520 4521 4522 4524 4525 4527 4528 4529 4530 4531 4532 4533 4535 4536 4537 4538 4539 4540 4541 4542 4543 4550 4551 4553 4554 4560 4561 4562 4564 4565 4566 4567 4568 4569 4570 4571 4574 4575 4576 4581 4583 4584 4585 4586 4587 4588 4589 4675 4690 4691 4693 4694 4695 4696 4697 4698',
'Zuid-Holland':     '1428 1431 1433 2121 2159 2160 2161 2162 2163 2166 2170 2171 2172 2180 2181 2182 2190 2191 2200 2201 2202 2203 2204 2210 2211 2212 2215 2216 2220 2221 2222 2223 2224 2225 2230 2231 2232 2235 2240 2241 2242 2243 2244 2245 2250 2251 2252 2253 2254 2260 2261 2262 2263 2264 2265 2266 2267 2270 2271 2272 2273 2274 2275 2280 2281 2282 2283 2284 2285 2286 2287 2288 2289 2290 2291 2292 2295 2300 2301 2302 2303 2311 2312 2313 2314 2315 2316 2317 2318 2321 2322 2323 2324 2331 2332 2333 2334 2340 2341 2342 2343 2350 2351 2352 2353 2355 2360 2361 2362 2370 2371 2374 2375 2376 2377 2380 2381 2382 2390 2391 2394 2396 2400 2401 2402 2403 2404 2405 2406 2407 2408 2409 2410 2411 2412 2415 2420 2421 2430 2431 2432 2435 2440 2441 2445 2450 2451 2460 2461 2465 2470 2471 2480 2481 2490 2491 2492 2493 2495 2496 2497 2498 2500 2501 2502 2503 2504 2505 2506 2507 2508 2509 2511 2512 2513 2514 2515 2516 2517 2518 2521 2522 2523 2524 2525 2526 2531 2532 2533 2541 2542 2543 2544 2545 2546 2547 2548 2551 2552 2553 2554 2555 2561 2562 2563 2564 2565 2566 2571 2572 2573 2574 2581 2582 2583 2584 2585 2586 2587 2591 2592 2593 2594 2595 2596 2597 2600 2601 2611 2612 2613 2614 2616 2622 2623 2624 2625 2626 2627 2628 2629 2630 2631 2632 2635 2636 2640 2641 2642 2643 2645 2650 2651 2652 2660 2661 2662 2665 2670 2671 2672 2673 2675 2676 2678 2680 2681 2684 2685 2690 2691 2692 2693 2694 2700 2701 2702 2711 2712 2713 2715 2716 2717 2718 2719 2721 2722 2723 2724 2725 2726 2727 2728 2729 2730 2731 2735 2740 2741 2742 2743 2770 2771 2800 2801 2802 2803 2804 2805 2806 2807 2808 2809 2810 2811 2820 2821 2825 2830 2831 2850 2851 2855 2860 2861 2865 2870 2871 2872 2900 2901 2902 2903 2904 2905 2906 2907 2908 2909 2920 2921 2922 2923 2924 2925 2926 2930 2931 2935 2940 2941 2950 2951 2952 2953 2954 2957 2959 2960 2961 2964 2965 2967 2968 2969 2970 2971 2973 2974 2975 2977 2980 2981 2982 2983 2984 2985 2986 2987 2988 2989 2990 2991 2992 2993 2994 2995 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3011 3012 3013 3014 3015 3016 3021 3022 3023 3024 3025 3026 3027 3028 3029 3031 3032 3033 3034 3035 3036 3037 3038 3039 3041 3042 3043 3044 3045 3046 3047 3050 3051 3052 3053 3054 3055 3056 3059 3061 3062 3063 3064 3065 3066 3067 3068 3069 3071 3072 3073 3074 3075 3076 3077 3078 3079 3081 3082 3083 3084 3085 3086 3087 3088 3089 3100 3101 3102 3109 3111 3112 3113 3114 3115 3116 3117 3118 3119 3121 3122 3123 3124 3125 3130 3131 3132 3133 3134 3135 3136 3137 3138 3140 3141 3142 3143 3144 3145 3146 3147 3150 3151 3155 3160 3161 3162 3165 3170 3171 3172 3176 3181 3190 3191 3192 3193 3194 3195 3196 3197 3198 3199 3200 3201 3202 3203 3204 3205 3206 3207 3208 3209 3211 3212 3214 3216 3218 3220 3221 3222 3223 3224 3225 3227 3230 3231 3232 3233 3234 3235 3237 3238 3240 3241 3243 3244 3245 3247 3248 3249 3250 3251 3252 3253 3255 3256 3257 3258 3260 3261 3262 3263 3264 3265 3267 3270 3271 3273 3274 3280 3281 3284 3286 3290 3291 3292 3293 3295 3297 3299 3300 3301 3311 3312 3313 3314 3315 3316 3317 3318 3319 3328 3329 3330 3331 3332 3333 3334 3335 3336 3340 3341 3342 3343 3344 3350 3351 3352 3353 3354 3355 3356 3360 3361 3362 3363 3364 3366 3370 3371 3372 3373 3380 3381 3465 3466 3651 3652 3653 4126 4128 4140 4141 4142 4143 4145 4163 4200 4201 4202 4203 4204 4205 4206 4207 4208 4209 4213 4221 4223 4225 4230 4231 4233 4235 4240 4241 4243 4245 4247 4284 6300',
}
for k, v in province.items(): 
  province[k] = v.split() 

cities = { 
'LEDEACKER': 'Noord-Brabant',
'NIEUWE PEKELA': 'Groningen',
'VEGHEL': 'Noord-Brabant',
'NOTTER': 'Overijssel',
'ALTEVEER GN': 'Groningen',
'OUDHEUSDEN': 'Noord-Brabant',
'DOEVEREN': 'Noord-Brabant',
'HEESBEEN': 'Noord-Brabant',
'HERPT': 'Noord-Brabant',
'HEDIKHUIZEN': 'Noord-Brabant',
'HEDIKHUIZEN GEM HEUSDEN': 'Noord-Brabant',
'AMSTERDAM': 'Noord-Holland',
'AMSTERDAM ZUIDOOST': 'Noord-Holland',
'OUDERKERK AAN DE AMSTEL': 'Noord-Holland',
'ALKMAAR': 'Noord-Holland',
'AKERSLOOT': 'Noord-Holland',
'HAARLEM': 'Noord-Holland',
'BARNEVELD': 'Gelderland',
'GELDERMALSEN': 'Gelderland',
'OOSTERHOUT GEM NIJMEGEN': 'Gelderland',
'BRUMMEN': 'Gelderland',
'ARNHEM': 'Gelderland',
'BREEDENBROEK': 'Gelderland',
'UGCHELEN': 'Gelderland',
'JONKERSLAN': 'Friesland',
'BOAZUM': 'Friesland',
'BRITSWERT': 'Friesland',
'WIUWERT': 'Friesland',
'RIEN': 'Friesland',
'LYTSEWIERRUM': 'Friesland',
'WOMMELS': 'Friesland',
'KUBAARD': 'Friesland', 
'WINSUM FR': 'Friesland',
'BAIJUM':  'Groningen',
'WAAKSENS GEM LITT': 'Friesland',
'WAAXENS HENN': 'Friesland',
'BEARS FR': 'Friesland',
'WAAKSENS GEM SWF': 'Friesland',
'EASTEREIN': 'Friesland',
'OOSTEREND': 'Noord-Holland',
'ITENS': 'Friesland',
'REAHUS': 'Friesland',
'HIDAARD': 'Friesland',
'HUNS': 'Friesland',
'BAARD': 'Friesland',
'EASTERLITTENS': 'Friesland',
'BAAIUM': 'Friesland',
'WJELSRYP': 'Friesland',
'SPANNUM': 'Friesland',
'HINNAARD': 'Friesland',
'EASTERWIERRUM': 'Friesland',
'MANTGUM': 'Friesland',
'JORWERT': 'Friesland',
'WEIDUM': 'Friesland',
'JELLUM': 'Friesland',
'HILAARD': 'Friesland',
'DUIVENDRECHT': 'Noord-Holland',
'WESTENDORP': 'Gelderland',
'IJMUIDEN': 'Noord-Holland',
'OOSTERHOUT': 'Noord-Brabant',
'LEONS': 'Friesland',
'ZOETERMEER': 'Zuid-Holland',
"'S-GRAVENHAGE": 'Zuid-Holland',
'KRIMPEN AAN DEN IJSSEL': 'Zuid-Holland',
'POORTUGAAL': 'Zuid-Holland',
'UTRECHT': 'Utrecht',
'LOENERSLOOT': 'Utrecht',
'BERGSCHENHOEK': 'Zuid-Holland',
'RENSWOUDE': 'Utrecht',
'VLEUTEN': 'Utrecht',
'DELFT': 'Zuid-Holland',
'ROTTERDAM': 'Zuid-Holland',
'S-GRAVENHAGE': 'Zuid-Holland',
'ZEIST': 'Utrecht',
'HELLEVOETSLUIS': 'Zuid-Holland',
}


def zip2province(city, code):
  for k, v in province.items(): 
    if code[:4] in v: return k
  #print('war> unknown zip code {}'.format(code))
  if city in cities:
    return cities[city]
  return 'NA'

## read data
years = list(range(2010, 2020))
firms = ('enexis', 'liander', 'stedin')
net_manager = {'enexis': 'Enexis B.V.', 'liander': 'Liander N.V.', 'stedin': 'Stedin'}

elec = []
for firm in firms:
  for year in years:
    if firm in ('enexis', 'liander'):
      path = 'Electricity/{}_electricity_0101{}.csv'.format(firm, year)
    else:
      path = 'Electricity/{}_electricity_{}.csv'.format(firm, year)
    print('inf> reading {}'.format(path))
    df = pd.read_csv(path)
    df['year'] = year
    df['net_manager'] = net_manager[firm]
    elec.append(df)

df = pd.concat(elec, sort=True)

## make some columns
df['smartmeters'] = df.smartmeter_perc/100*df.num_connections
df['in_house'] = (100.-df.delivery_perc)/100.*df.annual_consume
df['active_connections'] = df.perc_of_active_connections/100.*df.num_connections
df['province'] = df.apply(lambda x: zip2province(x['city'], x['zipcode_from']), axis=1)
# df.to_csv('df.csv')


## plot total consumption per year by provider
print('dbg> plot total consumption per year by provider')
data = df.groupby(['net_manager', 'year']).agg({'annual_consume': 'sum'})
for k, v in net_manager.items():
  plt.plot(data.xs(v, level='net_manager'), label=v, marker='o')
plt.xlabel('Year')
plt.ylabel('Energy consumption [kWh]')
plt.legend()
plt.ylim(ymin=0)
plt.tight_layout()


## plot smartmeter fraction per year by provider
print('dbg> plot smartmeter fraction per year by provider')
data = df.groupby(['net_manager', 'year']).agg({'smartmeters': 'sum', 'num_connections': 'sum'})
data = data.smartmeters/data.num_connections*100

for k, v in net_manager.items():
  plt.plot(data.xs(v, level='net_manager'), label=v, marker='o')
plt.xlabel('Year')
plt.ylabel('Fraction of smart-meters [%]')
plt.legend()
plt.ylim(ymin=0)
plt.tight_layout()


## in-house power generation
print('dbg> in-house power generation')
data = df.groupby(['net_manager', 'year']).agg({'annual_consume': 'sum', 'in_house': 'sum'})
data = data.in_house/data.annual_consume*100

for k, v in net_manager.items():
  plt.plot(data.xs(v, level='net_manager'), label=v, marker='o')
plt.xlabel('Year')
plt.ylabel('Fraction of in-house power generation [%]')
plt.legend()
plt.ylim(ymin=0)
plt.tight_layout()


## power consumption per connection
print('dbg> power consumption per connection')
data = df.groupby(['net_manager', 'year']).agg({'annual_consume': 'sum', 'active_connections': 'sum'})
data = data.annual_consume/data.active_connections

for k, v in net_manager.items():
  plt.plot(data.xs(v, level='net_manager'), label=v, marker='o')
plt.xlabel('Year')
plt.ylabel('Energy consumption per connection [kWh]')
plt.legend()
plt.ylim(ymin=0)
plt.tight_layout()


## netherland province boundaries
print('dbg> netherland province boundaries')
path = r'NLD_adm/NLD_adm1.shp'
nld = gpd.read_file(path)
nld = nld[nld.ENGTYPE_1!='Water body']
nld.plot()


## plot smartmeter fraction per province in 2019
print('dbg> plot smartmeter fraction per province in 2019')
data = df[df.year==2019].groupby(['province']).agg({'smartmeters': 'sum', 'num_connections': 'sum'})
data = data.smartmeters/data.num_connections*100

datum = pd.DataFrame(data)
datum['province'] = datum.index
datum.reset_index(drop=True, inplace=True)  
datum.columns = ['ratio', 'province']
gdf = gpd.GeoDataFrame(pd.merge(nld[['NAME_1','geometry']], datum, left_on='NAME_1', right_on='province'))
options = {'title': '[%]'}
fig = gv.Polygons(gdf, vdims=['NAME_1', 'ratio']).opts(
    tools=['hover'], width=500, height=400, color_index='ratio',
    colorbar=True, toolbar='above', xaxis=None, yaxis=None).opts(title="スマートメータ普及率 (2019)", colorbar_opts=options)
gv.save(fig, 'nld-fraction-smartmeter.html')


## in-house power generation
print('dbg> in-house power generation')
data = df[df.year==2019].groupby(['province']).agg({'annual_consume': 'sum', 'in_house': 'sum'})
data = data.in_house/data.annual_consume*100

datum = pd.DataFrame(data)
datum['province'] = datum.index
datum.reset_index(drop=True, inplace=True)  
datum.columns = ['ratio', 'province']
gdf = gpd.GeoDataFrame(pd.merge(nld[['NAME_1','geometry']], datum, left_on='NAME_1', right_on='province'))
options = {'title': '[%]'}
fig = gv.Polygons(gdf, vdims=['NAME_1', 'ratio']).opts(
    tools=['hover'], width=500, height=400, color_index='ratio',
    colorbar=True, toolbar='above', xaxis=None, yaxis=None).opts(title="自家発電率 (2019)", colorbar_opts=options)
gv.save(fig, 'nld-inhouse-generation.html')


## power consumption per connection
print('dbg> power consumption per connection')
data = df[df.year==2019].groupby(['province']).agg({'annual_consume': 'sum', 'active_connections': 'sum'})
data = data.annual_consume/data.active_connections

datum = pd.DataFrame(data)
datum['province'] = datum.index
datum.reset_index(drop=True, inplace=True)  
datum.columns = ['ratio', 'province']
gdf = gpd.GeoDataFrame(pd.merge(nld[['NAME_1','geometry']], datum, left_on='NAME_1', right_on='province'))
options = {'title': '[kWh]'}
fig = gv.Polygons(gdf, vdims=['NAME_1', 'ratio']).opts(
    tools=['hover'], width=500, height=400, color_index='ratio',
    colorbar=True, toolbar='above', xaxis=None, yaxis=None).opts(title="接続毎電力消費量 (2019)", colorbar_opts=options)
gv.save(fig, 'nld-consumption-connection.html')

# eof
