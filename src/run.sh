#!/bin/sh
python train.py --fold 0 --model xgb
python train.py --fold 1 --model xgb
python train.py --fold 2 --model xgb
python train.py --fold 3 --model xgb
python train.py --fold 4 --model xgb