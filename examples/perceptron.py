# from Grokking Deep Learning
# Benchmark : cargo build && time cargo run --example perceptron; sleep 10; time python3.6 perceptron.py     

# Result:
# Linux Kernel: 5.3.0-42-generic
# CPU: 12 × Intel® Core™ i7-9750H CPU @ 2.60GHz
# RAM: 15,5 Gio
# > Rust : 9,21s user 0,10s system 102% cpu 9,109 total
# > Python : 9,66s user 0,66s system 112% cpu 9,201 total

import numpy as np

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output>0

streetlights = np.array( [[ 1, 0, 1 ],
[ 0, 1, 1 ],
[ 0, 0, 1 ],
[ 1, 1, 1 ] ] )
walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T
alpha = 0.2
hidden_size = 4
weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

for iteration in range(100_000):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)
        layer_2_delta = (layer_2 - walk_vs_stop[i:i+1])
        layer_1_delta=layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)
        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

    if(iteration % 10 == 9):
        print("Error:" + str(layer_2_error))