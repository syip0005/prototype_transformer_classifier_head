#!/bin/bash
#SBATCH --job-name=NEWJOB

# Replace this with your email address
# To get email updates when your job starts, ends, or fails
#SBATCH --mail-user=syip0005@student.monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Replace <project> with your project ID
#SBATCH --account=az20

#SBATCH --time=24:00:00
#SBATCH --ntasks=6
#SBATCH --gres=gpu:2
#SBATCH --partition=m3g
#SBATCH --mem=55G

# Edit this section to activate your conda environment
source /home/scotty/az20_scratch/scotty/miniconda/bin/activate
conda activate pyt4

cd /home/scotty/az20_scratch/scotty/multilabelquery

python main_train.py --entity=MYNAME --dataset=AG_NEWS --model=bert_classifier --batch=16 --epochs=15 \
--lr=1e-6 --weight_decay=0 --bert_type=bert-base-uncased --loss=CrossEntropyLoss --learning_rate_style=constant \
--optimiser=Adam
EOF