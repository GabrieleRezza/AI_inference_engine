import struct
MAGIC_NUMBER = 0x1404A715
num_layers = 2
layer1_input_size = 2
layer1_output_size = 3
layer1_weights = [1, -2, 3, 4, 5, -6]
layer1_bias = [-3, 2, -0.5]
layer2_input_size = 3
layer2_output_size = 2
layer2_weights = [6, 5.5, 4, -3, 2, 1]
layer2_bias = [2, -2]

with open("weights_biases.bin", "wb") as f:
    f.write(struct.pack("i", MAGIC_NUMBER))    
    f.write(struct.pack("i", num_layers))
    f.write(struct.pack("i", layer1_input_size))
    f.write(struct.pack("i", layer1_output_size))
    f.write(struct.pack("i", layer2_input_size))
    f.write(struct.pack("i", layer2_output_size))
    f.write(struct.pack(f"{len(layer1_weights)}f", *layer1_weights))
    f.write(struct.pack(f"{len(layer1_bias)}f", *layer1_bias))
    f.write(struct.pack(f"{len(layer2_weights)}f", *layer2_weights))
    f.write(struct.pack(f"{len(layer2_bias)}f", *layer2_bias))

