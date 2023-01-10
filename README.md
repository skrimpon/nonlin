# Leveraging Deep Learning Methods for Function Inverse Problems in Communication Systems
---
* Panagiotis Skrimponis, NYU Tandon School of Engineering
* Mustafa F. &#214;zko&#231;, NYU Tandon School of Engineering
---

## Abstract &#x1F4D8;
Wireless communications over millimeter wave and terahertz 
frequency bands have attracted considerable attention for 
future cellular systems due to the abundantly available 
bandwidth. However, communications over these high frequencies 
are vulnerable to blockage, shadowing. Moreover, the high penetration loss in these frequencies degrades the system performance even further. To compensate for the harsh propagation characteristics, new antenna designs using large number of antennas are conceived. Combined with beamforming techniques, these new antenna designs can realize the full potential of these high frequencies. However, the received signal quality may significantly degrade due to hardware imperfections of the radio 
frequency front-end (RFFE) devices. Specifically, the limitations 
such as noise figure, non-linear distortion, and phase offsets 
introduced by the radio equipment can significantly deteriorate
the system performance. The problem is exacerbated when the input power is too low or too high due to the limited range of ADC.
The RFFE distortion can be represented as a
non-linear function where the input is the original signal and 
output is the distorted signal which generally results in information
loss. We make novel use of deep learning methods for function
inverse problems, at the distortion mitigation step of the
communication systems. Our results show, deep neural networks are promising tools to improve the communication quality in wireless networks suffering from RFFE distortions. We achieve up to 20dB increase in the SNR as compared to the baseline method.


## Full Report :books:
The rest of our full report can be found [here](https://github.com/skrimpon/nonlin/tree/main/docs/FinalReport.pdf). 


## Our Dataset is open :arrow_down:
The dataset can be found [here](https://drive.google.com/file/d/1cAaNZ0D9iEkazOOvfwzVQWmbXE1mrhBz/view?usp=sharing).

## For our training code :open_file_folder:
Our training code, including failed ones :broken_heart:, can be found [here](https://github.com/skrimpon/nonlin/tree/main/project/train).

## To reproduce our main result :framed_picture:
We provide the trained networks and the testing dataset in a colab notebook. We, also, committed this notebook in our github repository just so it would be here. However, please see the colab notebook [here](https://colab.research.google.com/drive/1Cg6ToHTp2Wmk7j7j_d9oViONBlHZD6K6?usp=sharing) for the best formatting and acutal reproduction of the results.

If you just want to read our results please see the notebook written as a markdown [here](https://github.com/skrimpon/nonlin/blob/main/demo/FinalResults.md). 

## Our main result :mortar_board:

>![Results](https://raw.githubusercontent.com/skrimpon/nonlin/main/performance_eval.png)
>
> The performance of our DNN compared with the state of the art baseline. We observe a performance improvement up to 20 dB compared to the baseline. More importantly, our model significantly improves the performance specifically in the non-linear region.
