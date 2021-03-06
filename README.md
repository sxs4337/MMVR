### SEMANTICALLY INVARIANT TEXT-TO-IMAGE GENERATION at ICIP 2018

http://www.cis.rit.edu/~sxs4337/publication/Sah_ICIP18_Semantically_Invariant.pdf

Dependencies--
Caffe LRCN- A clone from the recurrent branch of the LRCN caffe by Jeff Donahue http://jeffdonahue.com/lrcn/
Python 2.7

# Download the following three pre-trained models --

a. DeepSim image generator https://lmb.informatik.uni-freiburg.de/people/dosovits/code.html
   The baseline is the fc6 generator.
   Dosovitskiy, Alexey, and Thomas Brox. "Generating images with perceptual similarity metrics based on deep networks." Advances in Neural Information Processing Systems. 2016.
   Fine-tune it on MS-COCO dataset using instructions in directory "finetune_generator".
   
b. LRCN captioner http://www.cs.uwyo.edu/~anguyen8/share/lrcn_caffenet_iter_110000.caffemodel
   Nguyen, Anh, et al. "Plug & play generative networks: Conditional iterative generation of images in latent space." arXiv preprint arXiv:1612.00005 (2016).

c. CaffeNet https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
   Jia, Yangqing, et al. "Caffe: Convolutional architecture for fast feature embedding." Proceedings of the 22nd ACM international conference on Multimedia. ACM, 2014.

All trained models go in "nets" directory. Set the paths to models in settings.py, n_gram_condition.sh and multiple_caption_condition.sh files.

Run following commands on the shell scripts to change permissions.

chmod +x ./n_gram_condition.sh
chmod +x ./multiple_caption_condition.sh


# Experiments

1) ./n_gram_condition.sh a_pizza_on_a_table_at_a_restaurant

    The parameters are defined in n_gram_condition.sh script. n_gram options: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4".
    
2) ./multiple_caption_condition.sh input_captions.txt

   The input_captions.txt is a text file of captions that we would like to condition on.

## Citation
    @inproceedings{sah2018multimodal,
      title={Multimodal Reconstruction Using Vector Representation},
      author={Sah, Shagan and Shringi, Ameya and Peri, Dheeraj and Hamilton, John and Savakis, Andreas and Ptucha, Ray},
      booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
      pages={3763--3767},
      year={2018},
      organization={IEEE}
    }
