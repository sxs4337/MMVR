import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
import sys

site.addsitedir(settings.caffe_root)
pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
print pycaffe_root

import caffe
import numpy as np
import time
import os
import sys
import argparse # parsing arguments

#def get_blob_by_name(net, blob_name):
#  return net._blobs[net.net._blob_names_index[blob_name]]


def train(start_snapshot=0):

  max_iter = 1300000 # maximum number of iterations
  display_every = 50 # show losses every so many iterations
  snapshot_every = 10000 # snapshot every so many iterations
  snapshot_folder = 'snapshots_ImageNet' # where to save the snapshots (and load from)

  # we store a bunch of old generated images and feed them to the discriminator so that it does not overfit to the current data
  gen_history_size = 10000 # how many history images to store
  use_buffer_from = -1 # set to negative for not using the buffer
  gpu_id = 0
  feat_shape = (4096,)
  comparator_feat_shape = (256,6,6)
#  im_size = (3,227,227)
  batch_size = 64  # 64
  snapshot_at_iter = -1
  snapshot_at_iter_file = 'snapshot_at_iter.txt'

  sub_nets = ('encoder', 'generator', 'discriminator', 'data')

  if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)

  # make solvers
  with open ("solver_template.prototxt", "r") as myfile:
    solver_template=myfile.read()

  for curr_net in sub_nets:
    with open("solver_%s.prototxt" % curr_net, "w") as myfile:
      myfile.write(solver_template.replace('@NET@', curr_net))

  # initialize the nets
  caffe.set_mode_gpu()
  caffe.set_device(gpu_id)

  E = caffe.AdamSolver('solver_encoder.prototxt')
  G = caffe.AdamSolver('solver_generator.prototxt')
  D = caffe.AdamSolver('solver_discriminator.prototxt')
  data_reader = caffe.AdamSolver('solver_data.prototxt')
  # pdb.set_trace()

  # Load encoder and upconv
  # E.net.copy_from('encoder.caffemodel')
  encoder_model = 'bvlc_reference_caffenet.caffemodel'
  E.net.copy_from(encoder_model)

  # load from snapshot
  if start_snapshot > 0:
    curr_snapshot_folder = snapshot_folder +'/' + str(start_snapshot)
    print '\n === Starting from snapshot ' + curr_snapshot_folder + ' ===\n'
    generator_caffemodel = curr_snapshot_folder +'/' + 'generator.caffemodel'
    if os.path.isfile(generator_caffemodel):
      G.net.copy_from(generator_caffemodel)
    else:
      raise Exception('File %s does not exist' % generator_caffemodel)
    discriminator_caffemodel = curr_snapshot_folder +'/' + 'discriminator.caffemodel'
    if os.path.isfile(discriminator_caffemodel):
      D.net.copy_from(discriminator_caffemodel)
    else:
      raise Exception('File %s does not exist' % discriminator_caffemodel)

  # read weights of losses
  L_img_weight = G.net._blob_loss_weights[G.net._blob_names_index['img_recon_loss']]
  L_feat_weight = E.net._blob_loss_weights[E.net._blob_names_index['feat_recon_loss']] # Comparator
  D_loss_weight = D.net._blob_loss_weights[D.net._blob_names_index['softmax_loss']]

  train_D = True
  train_G = True

  # There are shortcuts so that you dont' have to specify layer names
  # forward_simple: forward from the first to the last layer
  # backward_simple: backward from the last to the first layer

  # do training
  start = time.time()
  for it in range(start_snapshot,max_iter):

    # Steps:
    # 1. Take x and push it through E to get h
    # 2. Push h through G to get x_hat
    # 3. Send both the x_hat and x to D and backprop to get the gradient to update D
    # 4. Push x_hat to D again to get the gradient to update G
    # 5. Update G with 3 losses: L = L_code + L_img + L_adv
    # 6. Update D with a single loss: L_disc

    # Get an image x from the dataset
    data_reader.net.forward_simple()

    # Push the image x through encoder E to get a real code h
    img_real = np.copy(data_reader.net.blobs['data'].data)
    E.net.blobs['data'].data[...] = img_real
    E.net.blobs['comparator_feat_in'].data[...] = np.zeros((batch_size,) + comparator_feat_shape, dtype='float32')

    # Get two features out of encoder: comparator feature (pool5) and optimization feature (fc6)
    E.net.forward_simple()
    feat_real = np.copy(E.net.blobs['feat'].data)
    comparator_feat_real = np.copy(E.net.blobs['comparator_feat'].data)

    # Push a real feature to G to get an image x_hat and a reconstruction loss in pixel space
    G.net.blobs['feat'].data[...] = feat_real
    G.net.blobs['data'].data[...] = img_real
    G.net.forward_simple()

    img_fake = G.net.blobs['generated_img'].data
    L_img = G.net.blobs['img_recon_loss'].data

    # Push the generated image through E to compute the feature loss
    E.net.blobs['data'].data[...] = img_fake
    E.net.blobs['comparator_feat_in'].data[...] = comparator_feat_real

    E.net.forward_simple()

    # Comparator loss in the feature (pool5)
    L_feat = E.net.blobs['feat_recon_loss'].data

    # Push real images to D
    D.net.blobs['data'].data[...] = img_real
    D.net.blobs['label'].data[...] = np.zeros((batch_size,1,1,1), dtype='float32')
    D.net.blobs['feat'].data[...] = feat_real

    D.net.forward_simple()
    loss_D_real = np.copy(D.net.blobs['softmax_loss'].data)

    # Compute the gradient of loss
    if train_D:
      D.increment_iter()
      D.net.clear_param_diffs()
      D.net.backward_simple()

    # Run D on the fake data
    D.net.blobs['data'].data[...] = img_fake
    D.net.blobs['label'].data[...] = np.ones((batch_size,1,1,1), dtype='float32')
    D.net.blobs['feat'].data[...] = feat_real

    D.net.forward_simple()

    loss_D_fake = np.copy(D.net.blobs['softmax_loss'].data)

    # Update D
    if train_D:
      # Q: there are 2 backward passes before this update
      # Does this update combine the gradients from these two passes?
      # Assume; YES (gradient accumulation)
      D.net.backward_simple()
      D.apply_update()

    # Compute the gradient for G
    # G maximizes the probablity that D makes a mistake
    # Here we run D on generated data with opposite (wrong) labels to get the gradient for G
    D.net.blobs['data'].data[...] = img_fake
    D.net.blobs['label'].data[...] = np.zeros((batch_size,1,1,1), dtype='float32')
    D.net.blobs['feat'].data[...] = feat_real

    D.net.forward_simple()
    # Losses
    loss_G = np.copy(D.net.blobs['softmax_loss'].data)
    loss_D = loss_D_real + loss_D_fake

    if train_G:
      G.increment_iter()
      G.net.clear_param_diffs()

      # Backpropagate the error from D and E
      # and give them to G to update G
      E.net.backward_simple()
      D.net.backward_simple()

      # The gradient from D now comes from 3 sources (as it has never cleared)
      G.net.blobs['generated_img'].diff[...] = E.net.blobs['data'].diff + D.net.blobs['data'].diff
      G.net.backward_simple()
      G.apply_update()

    # display
    if it % display_every == 0:
      print "[%s] Iteration %d: %f seconds" % (time.strftime("%c"), it, time.time()-start)
      print "  img loss: %e * %e = %f" % (L_img, L_img_weight, L_img * L_img_weight)
      print "  feat loss: %e * %e = %f" % (L_feat, L_feat_weight, L_feat * L_feat_weight) # comparator loss
      print "  D real loss: %e * %e = %f" % (loss_D_real, D_loss_weight, loss_D_real*D_loss_weight)
      print "  D fake loss: %e * %e = %f" % (loss_D_fake, D_loss_weight, loss_D_fake*D_loss_weight)
      print "  G loss: %e * %e = %f" % (loss_G, D_loss_weight, loss_G*D_loss_weight)
      start = time.time()
      if os.path.isfile(snapshot_at_iter_file):
        with open (snapshot_at_iter_file, "r") as myfile:
          snapshot_at_iter = int(myfile.read())

    # snapshot
    if it % snapshot_every == 0 or it == snapshot_at_iter:
      curr_snapshot_folder = snapshot_folder +'/' + str(it)
      print '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
      if not os.path.exists(curr_snapshot_folder):
        os.makedirs(curr_snapshot_folder)
      generator_caffemodel = curr_snapshot_folder + '/' + 'generator.caffemodel'
      G.net.save(generator_caffemodel)
      discriminator_caffemodel = curr_snapshot_folder + '/' + 'discriminator.caffemodel'
      D.net.save(discriminator_caffemodel)
      #if it >= use_buffer_from and use_buffer_from > 0:
        #gen_buffer_file = curr_snapshot_folder +'/' + 'gen_buffer.npy'
        #np.save(gen_buffer_file, gen_buffer.data)

    # switch optimizing discriminator and generator, so that neither of them overfits too much
    loss_ratio = loss_D / loss_G
    if loss_ratio < 1e-1 and train_D:
      train_D = False
      train_G = True
      print "<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_D=%d, train_G=%d >>>" \
            % (loss_D_real, loss_D_fake, loss_G, train_D, train_G)
    if loss_ratio > 5e-1 and not train_D:
      train_D = True
      train_G = True
      print " <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_D=%d, train_G=%d >>>" \
            % (loss_D_real, loss_D_fake, loss_G, train_D, train_G)
    if loss_ratio > 1e1 and train_G:
      train_G = False
      train_D = True
      print "<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_D=%d, train_G=%d >>>" \
            % (loss_D_real, loss_D_fake, loss_G, train_D, train_G)


  # This is here to show how to set weight losses
  # discriminator.net._blob_loss_weights[discriminator.net._blob_names_index['softmax_real']] = 100

def main():

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--start_snapshot', metavar='i', type=int, default=0, help='Resume from a snapshot')

  # parser.add_argument('--output_dir', metavar='b', type=str, default=".", help='Output directory for saving results')
  # parser.add_argument('--net_weights', metavar='b', type=str, default=settings.net_weights, help='Weights of the net being visualized')
  # parser.add_argument('--net_definition', metavar='b', type=str, default=settings.net_definition, help='Definition of the net being visualized')

  args = parser.parse_args()

  # Fix the seed
  # np.random.seed(args.seed)
  print args
  train(start_snapshot=args.start_snapshot)

if __name__ == '__main__':
  main()
