
# Take a text file of captions as input
if [ "$#" -ne "1" ]; then
  echo "Provide a file with input sentences."
  exit 1
fi

# Define hyper parameters
opt_layer=fc6     			# This is fixed to be fc6 unless we change the generator
act_layer=fc8    			# fc8 because the LRCN extract fc8 features from AlexNet
input_caption_file="${1}"   # A sentence with underscores between words e.g. a_pizza_on_a_table_at_a_restaurant 
xy=0             			# Spatial position for conv layers, for fc layers: xy = 0

n_iters=200       			# Run for N iterations
reset_every=0    			# Reset the code every N iterations (for diversity). 0 to disable resetting.
save_every=0      			# Save a sample every N iterations. 0 to disable saving intermediate samples.
lr=1              			# Initial learning rate 
lr_end=1e-10      			# Linearly decay toward this ending lr (e.g. for decaying toward 0, set lr_end = 1e-10)
threshold=0       			# Filter out samples below this threshold e.g. 0.98

# Weight scaling in the update equations as in PPGN
epsilon1=1e-3     			# prior
epsilon2=1        			# condition
epsilon3=1e-17    			# noise
# -----------------------------------------------

init_file="None"    		# Start from a random code

# Condition net
net_weights="nets/lrcn/lrcn_caffenet_iter_110000.caffemodel"
net_definition="nets/caffenet/caffenet.prototxt"
captioner_definition="nets/lrcn/lrcn_word_to_preds.deploy.prototxt"
#-----------------------

python ./sampling_caption_multi.py \
    --act_layer ${act_layer} \
    --opt_layer ${opt_layer} \
    --sentence ${sentence} \
    --xy ${xy} \
    --n_iters ${n_iters} \
    --save_every ${save_every} \
    --reset_every ${reset_every} \
    --lr ${lr} \
    --lr_end ${lr_end} \
    --seed ${seed} \
    --init_file ${init_file} \
    --epsilon1 ${epsilon1} \
    --epsilon2 ${epsilon2} \
    --epsilon3 ${epsilon3} \
    --threshold ${threshold} \
    --net_weights ${net_weights} \
    --net_definition ${net_definition} \
    --captioner_definition ${captioner_definition}\
    --input_caption_file ${input_caption_file}