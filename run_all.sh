CUDA_VISIBLE_DEVICES=0 python3 test.py --model_path ./previous/fcn8s_retinopathy_best_model.pkl  --dataset retinopathy --img_dir /mnt/data/datasets/ --out_path ./train-dataset/no-apparent-retinopathy/SE --arch fcn8s --suffix .jpg

CUDA_VISIBLE_DEVICES=0 python3 test.py --model_path fcn8s_retinopathy_HE_best_model.pkl  --dataset retinopathy --img_dir /mnt/data/datasets/ --out_path ./train-dataset/no-apparent-retinopathy/HE --arch fcn8s --suffix .jpg

CUDA_VISIBLE_DEVICES=0 python3 test.py --model_path fcn8s_retinopathy_best_model_MA.pkl  --dataset retinopathy --img_dir /mnt/data/datasets/ --out_path ./train-dataset/no-apparent-retinopathy/MA --arch fcn8s --suffix .jpg

CUDA_VISIBLE_DEVICES=0 python3 test.py --model_path fcn8s_retinopathy_EX_best_model.pkl  --dataset retinopathy --img_dir /mnt/data/datasets/ --out_path ./train-dataset/no-apparent-retinopathy/EX --arch fcn8s --suffix .jpg


