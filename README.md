# Original Code for Paper "Crossed-Time Delay Neural Network for Speaker Recognition"
 

The main model architecrue CTDNN is defined in `Models/model.py` . CTDNN can replace original TDNN easily in your own model. Note that the default parameters may not be equal to the papers'.

The dataset preprocess script and training scripts are listed in corresponding directory, they should be easy to use with minor custom change in path.

If you use this work or code, please kindly cite the following paper:
```bib
@inproceedings{chen_crossed-time_2021,
	location = {Cham},
	title = {Crossed-Time Delay Neural Network for Speaker Recognition},
	isbn = {978-3-030-67832-6},
	pages = {1--10},
	booktitle = {{MultiMedia} Modeling},
	publisher = {Springer International Publishing},
	author = {Chen, Liang and Liang, Yanchun and Shi, Xiaohu and Zhou, You and Wu, Chunguo},
	editor = {Lokoč, Jakub and Skopal, Tomáš and Schoeffmann, Klaus and Mezaris, Vasileios and Li, Xirong and Vrochidis, Stefanos and Patras, Ioannis},
	date = {2021}
}
```