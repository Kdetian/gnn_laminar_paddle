from network_utils import *
from training_utils import *
from data_utils import lift_drag
import paddle

best_model   = InvariantEdgeModel(edge_feature_dims, num_filters, initializer)
best_model.set_state_dict(paddle.load('./best_model/best_model_e396'))
#best_model.load_weights('./best_model/best_model_e713')

#nodes = pd.read_csv('../data/cylindre/nodes.csv')[['x', 'y', 'Object']].values.astype('float32')#, 'u_bc1', 'u_bc2', 'dist0']]
#flow  = pd.read_csv('../data/cylindre/flow.csv').values.astype('float32')
#edges = pd.read_csv('../data/cylindre/edges.csv').values
nodes = pd.read_csv('./data/data/dataset_shapes_gcnn/naca/nodes.csv')[['x', 'y', 'Object']].values.astype('float32')#, 'u_bc1', 'u_bc2', 'dist0']]
flow  = pd.read_csv('./data/data/dataset_shapes_gcnn/naca/flow.csv').values.astype('float32')
edges = pd.read_csv('./data/data/dataset_shapes_gcnn/naca/edges.csv').values


print('non-used nodes', np.setdiff1d(np.arange(nodes.shape[0]), np.unique(edges)))
### delete useless nodes
nodes = nodes[np.unique(edges),:]
flow = flow[np.unique(edges),:]

##  reset node index
_, edges = np.unique(edges, return_inverse=True)#return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式存储。
edges = np.reshape(edges, (-1,2))

# nodes = tf.convert_to_tensor(nodes, dtype=tf.dtypes.float32)
# edges = tf.convert_to_tensor(edges, dtype=tf.dtypes.int32)
# flow  = tf.convert_to_tensor(flow, dtype=tf.dtypes.float32)
nodes = paddle.to_tensor(nodes, dtype = 'float32')
edges = paddle.to_tensor(edges,dtype = 'int32')
flow  = paddle.to_tensor(flow,dtype = 'float32')



x = (nodes[:, 0:1] + 2.0) / 4.0  # 0 column
y = (nodes[:, 1:2] + 2.0) / 4.0  # 1 column
objet = paddle.reshape(nodes[:, -1],[-1, 1])  # the last column
# x = tf.math.divide(tf.math.subtract(nodes[:,0:1], -2.0), 4.0)#0 column
# y = tf.math.divide(tf.math.subtract(nodes[:,1:2], -2.0), 4.0)#1 column
# objet    = tf.reshape(nodes[:,-1], (-1,1))#the last column
nodes = paddle.concat([x,y,objet], axis = 1)

min_uvp = paddle.min(flow, axis=0)
max_uvp = paddle.max(flow, axis=0)

# max_values = paddle.zeros([3,],dtype = 'float32')
# min_values = paddle.zeros([3,],dtype = 'float32')

# max_values = paddle.max(paddle.stack([max_values, max_uvp],axis = 0), axis = 0)
# min_values = paddle.min(paddle.stack([min_values, min_uvp],axis = 0), axis = 0)

min_values =paddle.to_tensor([-0.13420522212982178, -0.830278217792511, -1.9049606323242188], dtype=paddle.float32)
max_values = paddle.to_tensor([1.4902634620666504, 0.799094557762146, 1.558414101600647], dtype=paddle.float32)
flow2 = paddle.divide(paddle.subtract(flow, min_values), paddle.subtract(max_values, min_values))
# min_values = tf.constant([-0.13420522212982178, -0.830278217792511, -1.9049606323242188], dtype=tf.dtypes.float32, shape=(3,))
# max_values = tf.constant([1.4902634620666504, 0.799094557762146, 1.558414101600647], dtype=tf.dtypes.float32, shape=(3,))
# flow2 = tf.math.divide(tf.math.subtract(flow, min_values), tf.math.subtract(max_values, min_values))

##### compute MAE
count = count_neighbour_edges(nodes, edges)
print('{} nodes, {} edges.'.format(nodes.numpy().shape[0], edges.numpy().shape[0]))
print(' ')

edge_features = paddle.mean(nodes[:,:3][edges], axis = 1)
#edge_features = tf.math.reduce_mean(tf.gather(nodes[:,:3], edges), 1)
pred = best_model(nodes[:,:3], edges, edge_features, count)
loss = loss_fn(pred, flow2)
print('The MAE on this shape is {}.'.format(float(loss)))
print(' ')


###### compute drag
elements= pd.read_csv('./data/data/dataset_shapes_gcnn/naca/elements.csv').values
_, elements = np.unique(elements, return_inverse=True)
elements = np.reshape(elements, (-1,3))


D1, L1 = lift_drag(nodes.numpy(), edges.numpy(), elements, flow.numpy(), 0.1)
D2, L2 = lift_drag(nodes.numpy(), edges.numpy(), elements, pred.numpy(), 0.1)
print(D1, D2)


##### save predicted velocity and pressure
pred = pred.numpy()
pred1 = pred[0:5,:]
pred2 = np.zeros((129,3))#59
pred3 = pred[5:,:]
pred = np.vstack([pred1, pred2, pred3])
np.savetxt('best_model/naca.csv', pred, delimiter=',', header='u,v,p', fmt='%1.16f', comments='')
