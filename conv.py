import numpy as np

'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''


class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
    self.filter_before_inverse = 0
    self.filter_after_inverse = 0

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input, inverse_filter = False):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input
    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))
    output_inverse_final = np.zeros((h - 2, w - 2, self.num_filters))
    output_half_final = np.zeros((h - 2, w - 2, self.num_filters))
    if not inverse_filter:
        for im_region, i, j in self.iterate_regions(input):
          output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
    else:
        # print(self.filters)
        # print('self.filters',self.filters)
        filters_copy = self.filters.copy()
        filters_half_copy = self.filters.copy()
        counter1 = 0
        for q in self.filters:
            counter2 = 0
            # print('counter1', counter1)
            for n in q:
                for k in range(len(n)):
                    # if counter1 ==0:
                        filters_copy[counter1][counter2][k] = n[len(n) - k - 1]
                        filters_half_copy[counter1][counter2][k] = n[len(n) - k - 1]
                counter2 += 1
            counter1 += 1
        # print('filters_copy', filters_copy)
        for im_region, i, j in self.iterate_regions(input):
          output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
          output_half_final[i, j] = np.sum(im_region * filters_half_copy, axis=(1, 2))
          output_inverse_final[i, j] = np.sum(im_region * filters_copy, axis=(1, 2))
          # for m in range(len(output_inverse_final[i, j])):
          #   if output_inverse_final[i, j][m]<output[i, j][m]:
          #       output_inverse_final[i,j][m] = output[i, j][m]
                # if output_inverse_final[i, j][m]<output_half_final[i, j][m]:
                #     output_inverse_final[i,j][m] = output_half_final[i, j][m]
        # print('output', output)
        print('filters_copy',filters_copy)
        print('self.filters', self.filters)
        output = output_inverse_final.copy()
        self.filter_before_inverse = self.filters
        self.filter_after_inverse = filters_copy
        # print('output_inverse_final',output)
    return output
  def return_filters(self):
      filter_before_inverse = self.filter_before_inverse
      filter_after_inverse = self.filter_after_inverse
      return filter_before_inverse, filter_after_inverse

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return None
