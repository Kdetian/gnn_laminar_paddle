# Custom imports
from data_utils import *
from training_utils import *
from log import logs

logs.info("Devices in use:")
if paddle.device.is_compiled_with_cuda():
    logs.info("Name: %s \n", paddle.device.cuda.get_device_name(0))
    #paddle.device.set_device("gpu:0")
else:
    logs.info("Name: %s", "cpu")
    paddle.device.set_device("cpu")

if __name__ == '__main__':

    start      = time.time()
    # load data set
    nodes_set, edges_set, flow_set = load_dataset(2000, normalize=True, do_read=True, dataset_source='./dataset/dataset_toUse.txt')  # Dictionary with 2000 sample
    nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid, nodes_set_test, edges_set_test, flow_set_test = split(nodes_set, edges_set, flow_set, train_ratio, valid_ratio)
    del nodes_set_test, edges_set_test, flow_set_test, nodes_set, edges_set, flow_set

    # declare a new model
    my_model = InvariantEdgeModel(edge_feature_dims, num_filters, initializer)




    training_loss, validation_loss = training_loop(my_model,
                                                   num_epochs,
                                                   nodes_set_train,
                                                   edges_set_train,
                                                   flow_set_train,
                                                   nodes_set_valid,
                                                   edges_set_valid,
                                                   flow_set_valid,
                                                   decay_factor,
                                                   batch_size,
                                                   initial_learning_rate)
    end = time.time()

    logs.info('Training finished in %s seconds \n', (end - start))
    np.savetxt('best_model/training_loss.csv', training_loss, delimiter=',')
    np.savetxt('best_model/validation_loss.csv', validation_loss, delimiter=',')

    logs.info('The minimum validation loss attained is : %s', min(validation_loss))
