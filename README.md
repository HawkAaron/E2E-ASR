## File description
* model.py: rnnt joint model
* train_rnnt.py: rnnt training script
* train_ctc.py: ctc acoustic model training script
* eval.py: rnnt & ctc decode
* DataLoader.py: kaldi feature loader

## Run
* Extract feature
link kaldi timit example dirs (`local` `steps` `utils` )
excute `run.sh` to extract 40 dim fbank feature
run `feature_transform.sh` to get 123 dim feature as described in Graves2013

* Train CTC acoustic model
```
python train_ctc.py --lr 1e-3 --bi --noise --out exp/ctc_bi_lr1e-3 --schedule
```

* Train RNNT joint model
```
python train_rnnt.py <parameters> --initam <path to best CTC model> --schedule
```

* Decode 
```
python eval.py <path to best model> --ctc --bi --beam 100
```

## Reference
* RNN Transducer (Graves 2012): [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* RNNT joint (Graves 2013): [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778 )
* (PyTorch End-to-End Models for ASR)[https://github.com/awni/speech]
* (A Fast Sequence Transducer Implementation with PyTorch Bindings)[https://github.com/awni/transducer]

## TODO
* CTC beam search