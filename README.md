# Quantum Parameter Adaptation

Official source code for paper 《A Quantum Circuit-Based Compression Perspective for Parameter-Efficient Learning》

--- 

Overall Architecture of our propsed QPA

![image](https://github.com/CYLphysics/QPA/blob/main/figure/flow_.png)

## Usage
### Installation of QPA environment
```bash
git clone https://github.com/CYLphysics/QPA.git
cd QPA 
conda create --name qpa python=3.9 -y 
conda activate qpa
pip install qiskit==0.45.0 qiskit_ibm_runtime==0.16.0 qiskit-aer==0.13.0 transformers==4.20.0 datasets
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum && pip install --editable . && cd ..
```
We use the [TorchQuantum](https://github.com/mit-han-lab/torchquantum/tree/main) as the computational backend of quantum simulation.

### Fine-tune GPT2 by using QPA to generate LoRA parameters
```bash
python train.py
```

## Citation
If you find this code or idea useful, please cite our work:
```bib
@inproceedings{
liu2025a,
title={A Quantum Circuit-Based Compression Perspective for Parameter-Efficient Learning},
author={Chen-Yu Liu and Chao-Han Huck Yang and Hsi-Sheng Goan and Min-Hsiu Hsieh},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=bB0OKNpznp}
}
```
