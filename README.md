# raflow模型的pytorch实现

参照官方代码[radflow](https://github.com/alasdairtran/radflow)以及自己阅读论文的理解，用pytorch实现了radflow。因为是按照自己的理解实现，代码肯定会有一些和源代码不一样的地方，如果有发现请告知我，感谢。

```
@InProceedings{Tran2021Radflow,
  author = {Tran, Alasdair and Mathews, Alexander and Ong, Cheng Soon and Xie, Lexing},
  title = {Radflow: A Recurrent, Aggregated, and Decomposable Model for Networks of Time Series},
  year = {2021},
  publisher = {Association for Computing Machinery},
  url = {https://doi.org/10.1145/3442381.3449945},
  booktitle = {Proceedings of The Web Conference 2021}
}
```

## 结构

主目录下model.py是模型代码，main是主函数。model目录是自己使用其他模块实现的一些代码。

使用以下命令运行：

```
python main.py
```

## 数据

数据使用的是PEMS数据

## 依赖

主要用到的依赖包

1.torch

2.h5py