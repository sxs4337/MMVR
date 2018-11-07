#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def main():
  path = "loss.txt"

  data = np.loadtxt(fname=path, delimiter=" ", unpack=True)

  print data.shape
  # print data[:, 0]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(data[0, :], data[1, :])
  ax.plot(data[0, :], data[2, :])
  ax.plot(data[0, :], data[3, :])
  ax.plot(data[0, :], data[4, :])
  ax.plot(data[0, :], data[5, :])
  # plt.show()

  ax.set_xlabel('Training iteration')
  ax.set_ylabel('Loss')

  plt.legend(labels=['recon loss', 'feat loss', 'D(x) real loss', 'D(x) fake loss', 'G(z) fake loss'])
  plt.title('Training GAN on fc6 codes from AlexNet')

  output_file = "%s/loss_plot.jpg" % os.getcwd()
  
  plt.tight_layout()  
  plt.savefig(output_file, dpi = 100, pad_inches=0.07,  bbox_inches='tight')
  plt.close()

  print "Saved to %s" % output_file

  return None

if __name__ == '__main__':
  main()
