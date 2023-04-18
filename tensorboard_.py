from keras.callbacks import TensorBoard


def tensorboard(directory, net):
    log_dir = directory + "/" + net
    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    print('Callback: ' + directory + '/' + net)
    return tb_callback
