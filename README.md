# SkrSkr
This is the repository for SkrSkr that won the second place in [the 57th IEEE/ACM Design Automation Conference System Design Contest (DAC-SDC)](https://dac-sdc-2020.groups.et.byu.net/doku.php). Designed by
> SkrSkr, Reconfigurable and Intelligent Computing Lab, ShanghaiTech University.

> Jiang Weixiong, Liu Xinzhe, Sun Hao, Li Rui, Luo Shaobo, Yu Heng, Ha Yajun*

> hayj@shanghaitech.edu.cn

![image](RICL.png)

Our design bases on [SkyNet](https://github.com/TomG008/SkyNet), the champion design of [the 56th IEEE/ACM Design Automation Conference System Design Contest (DAC-SDC)](http://www.cse.cuhk.edu.hk/~byu/2019-DAC-SDC/index.html). We deliver 73.13% Intersection over Union (IoU) and 52.5fps on Ultra96v2 and 57fps on Ultra96v1. Our contributions are as follows:
1. **One Shot Fully Integer Quantization**
- One Shot: Our method does not require fine-tuning and calibration set (but achieves higher accuracy with a calibration set consists of 8 images).
- Fully Integer: No fixed point or floating point operation during the inference process expect computing bounding box on CPU. 
- Loss Free: We achieve the same IoU (73.13%) as the floating point model on the hidden dataset.
- Lower Bit: Weights are quantized to int6 and activations are quantized to uint8.
2. **Accelerator Optimization**
- Doubled Parallelism: Thanks to fully integer operation and lower bit width (w6a8 vs w11a9), the parallelism of PWCONV1X1 is optimized to 512 and that of DWCONV3X3 is optimized to 288/5=57.6.
- All II=1: The original [reorg](https://github.com/TomG008/SkyNet/blob/master/FPGA/HLS/net_hls.cc#L626), [computing bbox](https://github.com/TomG008/SkyNet/blob/master/FPGA/HLS/net_hls.cc#L25), [load w1x1](https://github.com/TomG008/SkyNet/blob/master/FPGA/HLS/conv1x1.cc#L76) in [SkyNet](https://github.com/TomG008/SkyNet) is un-optimized. After our optimization, all the functions whose II can be equal to 1 equal to 1.
- RGB to RGBA: We expand the bus width of [load img](https://github.com/TomG008/SkyNet/blob/master/FPGA/HLS/net_hls.cc#L512) from 8 bits to 32 bits. Correspondingly, we load images in RGBA format so that it can be fed directly into the accelerator without any transposition.
- Frequency Boost: The system's critical path is halven due to the reduction of the data bus bit width from 512 to 256 bits, thus the accelerator can run at 333MHz.

With the above optimization, the cycles of SkyNet accelerator is reduced from 18.88M to 10.58M.

3. **System Optimization**
- Move Stitching to PL: Loading and stitching images consume lots of DDR bandwidth, so we move the image stitching from CPU to PL.
- Resize -> Converting Color Space: Performing image resize before converting color space also helps to save DDR bandwidth.
- Multi-Process: We assign two processes for loading and resizing images.

## Platform
[Xilinx Ultra96](https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html)

- We recommend you test on Ultra96v1 and Ultra96v2-I-G (delivered after Jan 2020), since Ultra96v2-G has a **poorly designed power system** and can only run at 280MHz. 
- To enable power monitoring on Ultra96v2, please reference [jgoeders/dac_sdc_2020](https://github.com/jgoeders/dac_sdc_2020/tree/master/support/measure_power).

## Software prerequisites
[Vivado Design Suite - HLx Editions](https://www.xilinx.com/products/design-tools/vivado.html#overview)\
[SDSoC](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/archive-sdsoc.html)\
[PyTorch](https://pytorch.org/)

## Directly run the demo on the FPGA

This example allows you to directly try our bitstream and weights by running over sample1000 dataset. The images are processed with a batch size of 4. The host code (run.py) runs on the embedded ARM core. It first loads the weight file (SkrSkr.bin), and then loads the binary file (SkrSkr.bit) to configure the FPGA  rogram logic. Then it activates the SkyNet IP to execute the inference of input images, and outputs the coordinates of detected bounding boxes. Finally it shows the total execution time (s) and energy consumption (J).
To run the demo:
```
$ cd ./Deploy/
$ sudo python3 run.py
```
You should be able to see outputs like:
```
Allocating memory done
Parameters loading done
Bitstream loaded
Loading images to DDR...
Start...
0 0.jpg [297, 389, 132, 237]
1 1.jpg [305, 343, 193, 246]
2 2.jpg [559, 573, 232, 253]
3 3.jpg [240, 261, 158, 216]
...
992 992.jpg [298, 335, 152, 205]
993 993.jpg [301, 317, 169, 201]
994 994.jpg [281, 310, 170, 210]
995 995.jpg [316, 410, 159, 294]
Detection finished

Total time: 15.036086797714233 s
Total energy: 133.613382133 J
Average_IoU: 0.730889077814
```
The system throughput (1000/15=66.6fps) can reach the accelerator throughput (66.6fps) if loading images and converting color space is done ahead of run. However, the rule tells that the runtime and energy values should include the entire execution sequence, so we also provide run_multiprocess.py.
```
$ sudo python3 run_multiprocess.py
```
You should be able to see outputs like:
```
Allocating memory done
Parameters loading done
Bitstream loaded
Start...
0 0.jpg [297, 389, 132, 237]
1 1.jpg [305, 343, 193, 246]
2 2.jpg [559, 573, 232, 253]
3 3.jpg [240, 261, 158, 216]
...
992 992.jpg [298, 335, 152, 205]
993 993.jpg [301, 317, 169, 201]
994 994.jpg [281, 310, 170, 210]
995 995.jpg [316, 410, 159, 294]
Detection finished

Total time: 17.55455732345581 s
Total energy: 162.26445346 J
Average_IoU: 0.730889077814
```
Loading 1000 images and converting color space with two process costs about 2.5s so the final system throughput on Ultra96v1 is 1000/17.5=57fps.

## Build the bitstream from scratch

In this work, the SkyNet FPGA implementation is written in C code. To deploy the SkyNet on FPGA, we go through four major steps:

0. **C simulation in g++*:** Debug the code with g++ first and make sure the C code generates the same result as the GPU.
1. **Vivado HLS:** The C code is synthesized by Vivado High Level Synthesis (HLS) tool to generate RTL (Verilog code) code, and exported as an HLS IP.
2. **Vivado:** The exported Verilog code is synthesized by Vivado to generate bitstream for FPGA configuration.
3. **Host:** Upload the generated bitstream file (.bit), the hardware description file (.hwh), and the weight file (.bin) generated by Vivado HLS to FPGA, and finish the host code running in the embedded ARM core (in Python or C).

Or you can build the accelerator using SDSoC:

4. **SDSoC:** We combine SDSoC with PYNQ framework, you can generate the bitstream using SDSoC and use the generated .elf file as well as run.py, easy and convenient, isn't it?

### 0. C simulation in g++
Debugging in g++ is much faster than in the Vivado HLS tool. You can set up the environment according to [HLS Arbitrary Precision Types Library](https://github.com/Xilinx/HLS_arbitrary_Precision_Types), that provides simulation code of HLS Arbitrary Precision Types.
```
cd C/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
ln -s ../blob
ln -s ../weight
make -j16
./SkyNet
```
You will get
```
init SkyNet 
load parameter
load image
SkyNet start
DWCONV1+PWCONV1
DWCONV2+PWCONV2
DWCONV3+PWCONV3
DWCONV4+PWCONV4
DWCONV5+PWCONV5
REORG+DWCONV6
PWCONV6
CONV13
SkyNet costs 45469077us
check SkyNet results 
pool1 correct 
...
conv12 correct 
img 0 xmin: 0.463551, ymin: 0.365162, xmax: 0.607856, ymax: 0.657431
img 1 xmin: 0.463551, ymin: 0.365162, xmax: 0.607856, ymax: 0.657431
img 2 xmin: 0.463551, ymin: 0.365162, xmax: 0.607856, ymax: 0.657431
img 3 xmin: 0.463551, ymin: 0.365162, xmax: 0.607856, ymax: 0.657431
```
### 1. Vivado HLS
There are typically three steps:

1. C code synthesis
2. C and Verilog co-simulation
3. Export RTL (Verilog/VHDL)

You may go through the Vivado HLS flow by running:
```
$ vivado_hls hls.tcl
```
The C code synthesis takes roughly 14 minutes while the RTL exportation takes 2 minutes (on Ryzen 3700X). The output of this step is an exported HLS IP, written in Verilog. Accroding to our experience, co-simulation is not compulsory for designs like this that do not involve custom stream interfaces.

### 2. Vivado
In this step we integrate the generated HLS IP into the whole system, and generate the bitstream (.bit) and the hardware configuration file (.hwh). You may go through the Vivado flow by running:
```
$ cd ../RTL
$ vivado -mode tcl -source rtl.tcl
```
In this configuration, the Zynq processor works under 333MHz. After running this script, the generation of bitstream (.bit) is not completed even though the script shows to be terminated. It takes serveral hours for bitstream generation, and you may observe the progress in vivado GUI.

### 3. Host
After generating the bitstream, the final step is to finish the host code running in the processing system, in this case the embedded ARM core. Usually it is written in C, but the [PYNQ framework](https://github.com/Xilinx/PYNQ) allows us to write in Python.

First, find the following three files to upload to the board (default name and path):
1. **ultra96v2_wrapper.bit** (RTL/RTL.runs/impl_1)
2. **ultra96v2.hwh** (RTL/RTL.srcs/sources_1/bd/ultra96v2/hw_handoff)
3. **SkyNet.bin** (C/weight)

Remember to rename the .bit and .hwh file to SkyNet.bit and SkyNet.hwh, or anything but need to be the same. Second, in the Python host file, allocate memory for weights, off-chip buffers, load parameters, download the overlay (.bit) to program the FPGA logic and specify the IP addresses. You may refer to the run.py. 

### 4. SDSoC
We go through the process of generating bitstreams in SDSoC and call the accelerator in PYNQ (you can also use the .elf file generated by SDSoC). If you are using ZCU104 or ZCU102, you can build the bitstream with SDSoC. **Remember to modify the HLS Interface pragma**, although SDSoC supports overriding SDS pragma with HLS interface pragma, there will be wiring error if multiple interfaces share the bus with `bundle`. For instance, bundling `img` and `fm` together helps to save 18 BRAMs:
```c
void SkyNet(ADT4* img, ADT32* fm, WDT32* weight, BDT16* biasm)
{
#pragma HLS INTERFACE m_axi depth=204800 port=img    offset=slave bundle=fm
#pragma HLS INTERFACE m_axi depth=628115 port=fm     offset=slave bundle=fm
```
However, this will cause wiring error, so you have to assign an independent bus for `img` in SDSoC:
```c
void SkyNet(ADT4* img, ADT32* fm, WDT32* weight, BDT16* biasm)
{
#pragma HLS INTERFACE m_axi depth=204800 port=img    offset=slave bundle=img
#pragma HLS INTERFACE m_axi depth=628115 port=fm     offset=slave bundle=fm
```
That's why you cannot build the bitstream using SDSoC if you are using Ultra96, Ultra96 does not have enough BRAM if the `img` and `fm` interface are implemented seperately. However, we still provide the SDSoC platform of Ultra96 in case you manage to reduce the BRAM consumption. Noting that the **bitstream, not the BOOT.bin and image.ub** generated with Ultra96v1 SDSoC platform can run on Ultra96v2, and vice versa.
- [How to build SDSoC platform for Ultra96](https://blog.csdn.net/lulugay/article/details/83661407)
- [xilinx-ultra96v2-sdsoc-v2019.1.bsp](https://download.csdn.net/download/lulugay/12265684)
- [Ultra96v2 SDSoC platform 2019.1](https://download.csdn.net/download/lulugay/12265457)
- [Ultra96v1 SDSoC platform 2018.2](https://download.csdn.net/download/lulugay/10761826)
