import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    filter_before_inverse, filter_after_inverse = conv.return_filters()
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(1):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
            print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
# print('\n--- Testing the CNN Normal---')
# loss = 0
# num_correct = 0
# for im, label in zip(test_images, test_labels):
#     _, l, acc = forward(im, label)
#     loss += l
#     num_correct += acc
#
# num_tests = len(test_images)
# print('Test Loss:', loss / num_tests)
# print('Test Accuracy:', num_correct / num_tests)


test_images_77_inverse = []
for test_images_element in test_images:
    test_images_77_inverse_element_holder_final = []
    for k in test_images_element:
        test_images_77_inverse_element_holder = []
        for i in range(len(k)):
#             print(k[len(k)-1-i])
            test_images_77_inverse_element_holder.append(k[len(k)-1-i])
#         print(test_images_77_inverse_element_holder)
        test_images_77_inverse_element_holder_final.append(test_images_77_inverse_element_holder)
    test_images_77_inverse.append(test_images_77_inverse_element_holder_final)
test_images_77_inverse = np.asarray(test_images_77_inverse)



mid = round(len(test_images[77][0])/2)

test_images_half = []
for test_images_element in test_images:
    test_images_77_inverse_element_holder_final = []
    for k in test_images_element:
        test_images_77_inverse_element_holder = [0]*len(k)
        for i in range(round(len(k)/4)):
            test_images_77_inverse_element_holder[mid+i] = k[mid+2*i]
            test_images_77_inverse_element_holder[mid-i] = k[mid-2*i]
        test_images_77_inverse_element_holder_final.append(test_images_77_inverse_element_holder)
        test_images_half.append(test_images_77_inverse_element_holder_final)
test_images_half = np.asarray(test_images_half)

def test(type_data,type_filter, test_data):
    print('\n--- Testing the CNN '+type_data+' Data, '+type_filter+ ' filter ---')
    loss = 0
    num_correct = 0
    holder0 = 0
    holder1 = 0
    holder2 = 0
    holder3 = 0
    holder4 = 0
    holder5 = 0
    holder6 = 0
    holder7 = 0
    holder8 = 0
    holder9 = 0

    holder0correct = 0
    holder1correct = 0
    holder2correct = 0
    holder3correct = 0
    holder4correct = 0
    holder5correct = 0
    holder6correct = 0
    holder7correct = 0
    holder8correct = 0
    holder9correct = 0
    Print = 0
    for im, label in zip(test_data, test_labels):
        if type_filter =='normal':
            _, l, acc = forward(im, label)
        else:
            if Print<1:
                print('........................................')
                print('label', label)
                Print += 1
                _, l, acc = forward_inverse(im, label, printoutput = True)
                print('........................................')
                print()
                print()
            else:
                _, l, acc = forward_inverse(im, label)
        if label ==0:
            holder0 +=1
        elif label ==1:
            holder1 +=1
        elif label ==2:
            holder2 +=1
        elif label ==3:
            holder3 +=1
        elif label ==4:
            holder4 +=1
        elif label ==5:
            holder5 +=1
        elif label ==6:
            holder6 +=1
        elif label ==7:
            holder7 +=1
        elif label ==8:
            holder8 +=1
        elif label ==9:
            holder9 +=1


        if label ==0:
            if acc == 1:
                holder0correct +=1
        elif label ==1:
            if acc == 1:
                holder1correct +=1
        elif label ==2:
            if acc == 1:
                holder2correct +=1
        elif label ==3:
            if acc == 1:
                holder3correct +=1
        elif label ==4:
            if acc == 1:
                holder4correct +=1
        elif label ==5:
            if acc == 1:
                holder5correct +=1
        elif label ==6:
            if acc == 1:
                holder6correct +=1
        elif label ==7:
            if acc == 1:
                holder7correct +=1
        elif label ==8:
            if acc == 1:
                holder8correct +=1
        elif label ==9:
            if acc == 1:
                holder9correct +=1

        loss += l
        num_correct += acc

    # print('Test Accuracy holder0:', holder0correct / holder0)
    # print('Test Accuracy holder1:', holder1correct / holder1)
    # print('Test Accuracy holder2:', holder2correct / holder2)
    # print('Test Accuracy holder3:', holder3correct / holder3)
    # print('Test Accuracy holder4:', holder4correct / holder4)
    # print('Test Accuracy holder5:', holder5correct / holder5)
    # print('Test Accuracy holder6:', holder6correct / holder6)
    # print('Test Accuracy holder7:', holder7correct / holder7)
    # print('Test Accuracy holder8:', holder8correct / holder8)
    # print('Test Accuracy holder9:', holder9correct / holder9)

    num_tests = len(test_data)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)


import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
# from conv import inverse_filter
from softmax import Softmax
def forward_inverse(image, label, printoutput = False):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.

    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)
    if printoutput:
        print('.............normal filter.......................')
        print('out',out)

    out = conv.forward((image / 255) - 0.5, inverse_filter =True)
    out = pool.forward(out)
    out = softmax.forward(out)
    if printoutput:
        print('...............combined filter.....................')
        print('out',out)


    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc
#
# test('normal','normal', test_images)
# test('inverse','normal', test_images_77_inverse)
# test('normal','combined', test_images)
# test('Normal half ','combined', test_images_half)
# test('Normal half','normal', test_images_half)
test('inverse','combined', [test_images_77_inverse[77]])

print('....the thing below using the normal data')
test('fake_normal','fake_normal', [test_images[77]])
