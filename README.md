# AAAI19-TSLM
Source code for aaai19 "A Generalized Language Model in Tensor Space". AAAI-2019

## DEPENDENCIES
- python 2.7+
- numpy 1.13+
- tensorflow 1.2+
- torch 0.2+

## RUN
For TSLM, run:
```
python Main.py
```
For TSLM+MOS, run:
```
python main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15  --save PTB 
```

## REFERENCES
1. Yang, Z.; Dai, Z.; Salakhutdinov, R.; and Cohen, W. W. 2018. Breaking the softmax bottleneck: A high-rank RNN language model. In International Conference on Learning Representations(ICLR).
2. Mikolov T, Karafiát M, Burget L, et al. Recurrent neural network based language model. Eleventh Annual Conference of the International Speech Communication Association. 2010.
