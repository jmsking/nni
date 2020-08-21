# 搜索空间集合

## DartsCell

DartsCell 是从[这里](https://github.com/microsoft/nni/tree/master/examples/nas/darts)的 [CNN 模型](./DARTS.md)中提取出来的。 一个 DartsCell 是一个包含 N 个节点的序列的有向无环图 ，其中每个节点代表一个潜在特征的表示（例如卷积网络中的特征图）。 从节点1到节点2的有向边表示一些将节点1转换为节点2的操作。这些操作获取节点1的值并将转换的结果储存在节点2上。 节点之间的[操作](#darts-predefined-operations)是预定义的且不可更改。 一条边表示从预定义的操作中选择的一项，并将该操作将应用于边的起始节点。 一个 cell 包括两个输入节点，一个输出节点和其他 `n_node` 个节点。 输入节点定义为前两个 cell 的输出。 Cell 的输出是通过对所有中间节点进行归约运算（例如连接）而获得的。 为了使搜索空间连续，在所有可能的操作上通过softmax对特定操作选择进行松弛。 通过调整每个节点上softmax的权重，选择概率最高的操作作为最终结构的一部分。 可以通过堆叠多个cell组成一个CNN模型，从而构建一个搜索空间。 值得注意的是，在DARTS论文中，模型中的所有cell都具有相同的结构。

Darts的搜索空间如下图所示。 请注意，在实现中NNI将最后一个中间节点与输出节点进行了合并。

![](../../img/NAS_Darts_cell.svg)

预定义的操作在[参考](#predefined-operations-darts)中列出。

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.DartsCell
    :members:
```

### 示例代码

[示例代码](https://github.com/microsoft/nni/tree/master/examples/nas/search_space_zoo/darts_example.py)

```bash
git clone https://github.com/Microsoft/nni.git
cd nni/examples/nas/search_space_zoo
# search the best structure
python3 darts_example.py
```

<a name="predefined-operations-darts"></a>

### 参考

所有Darts支持的操作如下。

* MaxPool / AvgPool
  * MaxPool: Call `torch.nn.MaxPool2d`. This operation applies a 2D max pooling over all input channels. Its parameters `kernel_size=3` and `padding=1` are fixed. The pooling result will pass through a BatchNorm2d then return as the result.
  * AvgPool: Call `torch.nn.AvgPool2d`. This operation applies a 2D average pooling over all input channels. Its parameters `kernel_size=3` and `padding=1` are fixed. The pooling result will pass through a BatchNorm2d then return as the result.

    MaxPool / AvgPool with `kernel_size=3` and `padding=1` followed by BatchNorm2d
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.darts_ops.PoolBN
    ```
* SkipConnect

    There is no operation between two nodes. Call `torch.nn.Identity` to forward what it gets to the output.
* Zero operation

    There is no connection between two nodes.
* DilConv3x3 / DilConv5x5

    <a name="DilConv"></a>DilConv3x3: (Dilated) depthwise separable Conv. It's a 3x3 depthwise convolution with `C_in` groups, followed by a 1x1 pointwise convolution. It reduces the amount of parameters. Input is first passed through relu, then DilConv and finally batchNorm2d. **Note that the operation is not Dilated Convolution, but we follow the convention in NAS papers to name it DilConv.** 3x3 DilConv has parameters `kernel_size=3`, `padding=1` and 5x5 DilConv has parameters `kernel_size=5`, `padding=4`.
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.darts_ops.DilConv
    ```
* SepConv3x3 / SepConv5x5

    Composed of two DilConvs with fixed `kernel_size=3`, `padding=1` or `kernel_size=5`, `padding=2` sequentially.
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.darts_ops.SepConv
    ```

## ENASMicroLayer

This layer is extracted from the model designed [here](https://github.com/microsoft/nni/tree/master/examples/nas/enas). A model contains several blocks that share the same architecture. A block is made up of some normal layers and reduction layers, `ENASMicroLayer` is a unified implementation of the two types of layers. The only difference between the two layers is that reduction layers apply all operations with `stride=2`.

ENAS Micro employs a DAG with N nodes in one cell, where the nodes represent local computations, and the edges represent the flow of information between the N nodes. One cell contains two input nodes and a single output node. The following nodes choose two previous nodes as input and apply two operations from [predefined ones](#predefined-operations-enas) then add them as the output of this node. For example, Node 4 chooses Node 1 and Node 3 as inputs then applies `MaxPool` and `AvgPool` on the inputs respectively, then adds and sums them as the output of Node 4. Nodes that are not served as input for any other node are viewed as the output of the layer. If there are multiple output nodes, the model will calculate the average of these nodes as the layer output.

One structure in the ENAS micro search space is shown below.

![](../../img/NAS_ENAS_micro.svg)

The predefined operations can be seen [here](#predefined-operations-enas).

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.ENASMicroLayer
    :members:
```

The Reduction Layer is made up of two Conv operations followed by BatchNorm, each of them will output `C_out//2` channels and concat them in channels as the output. The Convolution has `kernel_size=1` and `stride=2`, and they perform alternate sampling on the input to reduce the resolution without loss of information. This layer is wrapped in `ENASMicroLayer`.

### Example code

[example code](https://github.com/microsoft/nni/tree/master/examples/nas/search_space_zoo/enas_micro_example.py)

```bash
git clone https://github.com/Microsoft/nni.git
cd nni/examples/nas/search_space_zoo
# search the best cell structure
python3 enas_micro_example.py
```

<a name="predefined-operations-enas"></a>

### References

All supported operations for ENAS micro search are listed below.

* MaxPool / AvgPool
    * MaxPool: Call `torch.nn.MaxPool2d`. This operation applies a 2D max pooling over all input channels followed by BatchNorm2d. Its parameters are fixed to `kernel_size=3`, `stride=1` and `padding=1`.
    * AvgPool: Call `torch.nn.AvgPool2d`. This operation applies a 2D average pooling over all input channels followed by BatchNorm2d. Its parameters are fixed to `kernel_size=3`, `stride=1` and `padding=1`.
    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.Pool
    ```

* SepConv
    * SepConvBN3x3: ReLU followed by a [DilConv](#DilConv) and BatchNorm. Convolution parameters are `kernel_size=3`, `stride=1` and `padding=1`.
    * SepConvBN5x5: Do the same operation as the previous one but it has different kernel sizes and paddings, which is set to 5 and 2 respectively.

    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.SepConvBN
    ```

* SkipConnect

    Call `torch.nn.Identity` to connect directly to the next cell.

## ENASMacroLayer

In Macro search, the controller makes two decisions for each layer: i) the [operation](#macro-operations) to perform on the result of the previous layer, ii) which the previous layer to connect to for SkipConnects. ENAS uses a controller to design the whole model architecture instead of one of its components. The output of operations is going to concat with the tensor of the chosen layer for SkipConnect. NNI provides [predefined operations](#macro-operations) for macro search, which are listed in [references](#macro-operations).

Part of one structure in the ENAS macro search space is shown below.

![](../../img/NAS_ENAS_macro.svg)

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.ENASMacroLayer
    :members:
```

To describe the whole search space, NNI provides a model, which is built by stacking the layers.

```eval_rst
..  autoclass:: nni.nas.pytorch.search_space_zoo.ENASMacroGeneralModel
    :members:
```

### Example code

[example code](https://github.com/microsoft/nni/tree/master/examples/nas/search_space_zoo/enas_macro_example.py)

```bash
git clone https://github.com/Microsoft/nni.git
cd nni/examples/nas/search_space_zoo
# search the best cell structure
python3 enas_macro_example.py
```

<a name="macro-operations"></a>

### References

All supported operations for ENAS macro search are listed below.

* ConvBranch

    All input first passes into a StdConv, which is made up of a 1x1Conv followed by BatchNorm2d and ReLU. Then the intermediate result goes through one of the operations listed below. The final result is calculated through a BatchNorm2d and ReLU as post-procedure.
    * Separable Conv3x3: If `separable=True`, the cell will use [SepConv](#DilConv) instead of normal Conv operation. SepConv's `kernel_size=3`, `stride=1` and `padding=1`.
    * Separable Conv5x5: SepConv's `kernel_size=5`, `stride=1` and `padding=2`.
    * Normal Conv3x3: If `separable=False`, the cell will use a normal Conv operations with `kernel_size=3`, `stride=1` and `padding=1`.
    * Normal Conv5x5: Conv's `kernel_size=5`, `stride=1` and `padding=2`.

    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.ConvBranch
    ```
* PoolBranch

    All input first passes into a StdConv, which is made up of a 1x1Conv followed by BatchNorm2d and ReLU. Then the intermediate goes through pooling operation followed by BatchNorm.
    * AvgPool: Call `torch.nn.AvgPool2d`. This operation applies a 2D average pooling over all input channels. Its parameters are fixed to `kernel_size=3`, `stride=1` and `padding=1`.
    * MaxPool: Call `torch.nn.MaxPool2d`. This operation applies a 2D max pooling over all input channels. Its parameters are fixed to `kernel_size=3`, `stride=1` and `padding=1`.

    ```eval_rst
    ..  autoclass:: nni.nas.pytorch.search_space_zoo.enas_ops.PoolBranch
    ```

<!-- push -->
