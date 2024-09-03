# PIE_Normal 96.3235
#python train_script.py --dataset PIE --model-path pie_normal --lr 0.001

# PIE_Conflict 87.5
#python train_script.py --dataset PIE --model-path pie_conflict --batch-size 200 --add-conflict

# HandWritten_normal 99.25
#python train_script.py --dataset HandWritten --model-path handwritten  --lr 0.001

# HandWritten_conflict 97.00
#python train_script.py --dataset HandWritten --model-path handwritten --batch-size 200  --add-conflict

# Scene_Normal 75.03
# python train_script.py --dataset Scene --model-path scene  --epochs 600  --annealing_step 100

# Scene_Conflict 66.44
# python train_script.py --dataset Scene --model-path scene  --epochs 600  --annealing_step 100 --add-conflict