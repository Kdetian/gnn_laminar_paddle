import math
from random import shuffle

from data_utils import *
from network_utils import *
from params import *
from log import logs


def count_neighbour_edges(node_features, edges):
    """
    return the degree of the nodes
    """
    n_nodes = node_features.shape[0]
    n_edges = edges.shape[0]

    ones = paddle.ones([n_edges, 1])

    count = paddle.slice(paddle.add(paddle.index_add_(paddle.zeros([n_edges, 1]), edges[:, 0],0, ones),
                      paddle.index_add_(paddle.zeros([n_edges, 1]), edges[:, 1],0, ones)), axes=[0, 1], starts=[0, 0], ends=[n_nodes, 1])

    # count      = torch.add(tf.math.unsorted_segment_sum(ones, edges[:, 0], n_nodes),
    #                          tf.math.unsorted_segment_sum(ones, edges[:, 1], n_nodes))
    return count


def mean_absolute_error(x):
    return paddle.mean(paddle.sum(paddle.abs(x), axis=0))


def loss_fn(prediction, real):
    return mean_absolute_error(paddle.subtract(prediction, real))


def watch_loss(model, nodes_set, edges_set, flow_set, do_batch=False, size_batch=64):
    """
    return the loss value of a mini-batch
    """

    shape_list = list(nodes_set.keys())
    num_graphs = len(shape_list)
    # Whether to batch or not
    if not do_batch:
        size_batch = num_graphs

    num_batch = math.ceil(num_graphs / size_batch)

    loss_of_epoch = 0.0
    total_nodes = 0.0
    for index in np.arange(num_batch):
        nodes_batch, edges_batch, flow_batch = get_batch_from_training_set(index, nodes_set, edges_set, flow_set,
                                                                           shape_list, size_batch)

        #logs.info(nodes_batch)
        count = count_neighbour_edges(nodes_batch, edges_batch)  # determine the degree of every node

        edge_features = paddle.mean(nodes_batch[:, :3][edges_batch], axis = 1)
        with paddle.no_grad():
            outputs = model(nodes_batch[:, :3], edges_batch, edge_features, count)
            loss = loss_fn(outputs, flow_batch)

        n_nodes_batch = nodes_batch.shape[0]
        loss_of_epoch += loss * n_nodes_batch  # Multiply loss by number of nodes in each batch graph
        total_nodes += n_nodes_batch

    return loss_of_epoch / total_nodes/159672/3


def train_model(model, epoch, nodes_set, edges_set, flow_set,shape_list, num_batch, size_batch):
    loss_of_epoch = 0.0
    total_nodes = 0.0
    for index in np.arange(num_batch):
        nodes_batch, edges_batch, flow_batch = get_batch_from_training_set(index, nodes_set, edges_set, flow_set,
                                                                           shape_list, size_batch)
        count = count_neighbour_edges(nodes_batch, edges_batch)  # degree of every node = number of neighboring nodes


        edge_features = paddle.mean(nodes_batch[:, :3][edges_batch], axis=1)  # Compute the edge features

        outputs = model(nodes_batch, edges_batch, edge_features, count)
        loss_batch = loss_fn(outputs, flow_batch)


        learning_rate = initial_learning_rate / (1.0 + decay_factor * epoch)

        optim = paddle.optimizer.Adam(learning_rate=learning_rate,
                                          parameters=model.parameters())# weight_decay=5e-4
        # compute gradient and do loss step
        loss_batch.backward()
        optim.step()
        optim.clear_grad()
        n_nodes_batch = nodes_batch.shape[0]
        loss_of_epoch += loss_batch * n_nodes_batch  # Multiply loss by number of nodes in each batch graph
        total_nodes += n_nodes_batch
        del optim,loss_batch, outputs

    return loss_of_epoch / total_nodes /(total_nodes / num_batch * 3) # Divided by the total number of nodes from all batch graphs


def training_loop(model, epochs_num, nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid,
                  flow_set_valid, decayfactor, size_batch,initial_rate):
    training_loss = list()
    validation_loss = [1000000000000000000.0]

    shape_list = list(nodes_set_train.keys())
    num_graphs = len(shape_list)
    num_batch = math.ceil(num_graphs / size_batch)  # divide the data set into num_batch mini-batches

    early_stop = 0

    for epoch in range(epochs_num):
        logs.info('Started epoch %s', epoch)
        shuffle.shuffle(shape_list)
        start = time.time()
        # learning_rate = initial_rate / (1.0 + decayfactor * epoch)
        # optim.lr.assign(learning_rate)

        train_loss = train_model(model,epoch, nodes_set_train, edges_set_train, flow_set_train,
                                 shape_list, num_batch, size_batch)


        training_loss.append(train_loss)

        valid_loss = watch_loss(model, nodes_set_valid, edges_set_valid, flow_set_valid, size_batch)
        validation_loss.append(valid_loss)

        # scheduler.step()

        if epoch == 0:
            # # 计算模型的参数总数量和训练参数数量
            total_params = model.parameters()
            trainable_params = []
            for param in total_params:
                if param.trainable:
                    trainable_params.append(param)
            logs.info("Total parameters: %s Trainable parameters: %s ",len(total_params),len(trainable_params))

        end = time.time()

        logs.info('Epoch %s: %s seconds --- training loss is %s --- validation loss is %s; \n', epoch, (end - start),
                  (train_loss.numpy()), (valid_loss.numpy()))

        if valid_loss < min(validation_loss[:-1]):
            early_stop = 0
            paddle.save(model.state_dict(), './best_model/best_model_e{}.pdparams'.format(epoch))
        else:
            early_stop += 1
        if early_stop == 500:
            break
    return training_loss, validation_loss[1:]
