from network_utils import *
from training_utils import *
from data_utils import lift_drag
import matplotlib.pyplot as plt
import paddle

best_model   = InvariantEdgeModel(edge_feature_dims, num_filters, initializer)

if os.path.exists('../data/data/dataset_shapes_gcnn') == False:
    dataset_location = '../data/data193599/dataset_shapes_gcnn.tar'
    t = tarfile.open(dataset_location)
    t.extractall(path='../data/data')
    t.close()

nodes = pd.read_csv('../data/data/dataset_shapes_gcnn/cylindre/nodes.csv')[['x', 'y', 'Object']].values.astype('float32')#, 'u_bc1', 'u_bc2', 'dist0']]
flow  = pd.read_csv('../data/data/dataset_shapes_gcnn/cylindre/flow.csv').values.astype('float32')
edges = pd.read_csv('../data/data/dataset_shapes_gcnn/cylindre/edges.csv').values

print('non-used nodes', np.setdiff1d(np.arange(nodes.shape[0]), np.unique(edges)))
### delete useless nodes
nodes = nodes[np.unique(edges),:]
flow = flow[np.unique(edges),:]

##  reset node index
_, edges = np.unique(edges, return_inverse=True)#return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式存储。
edges = np.reshape(edges, (-1,2))

nodes = paddle.to_tensor(nodes, dtype = 'float32')
edges = paddle.to_tensor(edges,dtype = 'int32')
flow  = paddle.to_tensor(flow,dtype = 'float32')

X= nodes[:,0].numpy()
Y = nodes[:, 1].numpy()


x = (nodes[:, 0:1] + 2.0) / 4.0  # 0 column
y = (nodes[:, 1:2] + 2.0) / 4.0  # 1 column
objet = paddle.reshape(nodes[:, -1],[-1, 1])  # the last column

nodes = paddle.concat([x,y,objet], axis = 1)


min_values =paddle.to_tensor([-0.13420522212982178, -0.830278217792511, -1.9049606323242188], dtype=paddle.float32)
max_values = paddle.to_tensor([1.4902634620666504, 0.799094557762146, 1.558414101600647], dtype=paddle.float32)
flow2 = paddle.divide(paddle.subtract(flow, min_values), paddle.subtract(max_values, min_values))


##### compute MAE
count = count_neighbour_edges(nodes, edges)
print('{} nodes, {} edges.'.format(nodes.numpy().shape[0], edges.numpy().shape[0]))
print(' ')

edge_features = paddle.mean(nodes[:,:3][edges], axis = 1)
#edge_features = tf.math.reduce_mean(tf.gather(nodes[:,:3], edges), 1)


no_use = best_model(nodes[:,:3], edges, edge_features, count)

loaded_params = paddle.load("./best_model/best_model_e991.pdparams")
best_model.set_state_dict(loaded_params)


pred = best_model(nodes[:,:3], edges, edge_features, count)
loss = loss_fn(pred, flow2)
print('The MAE on this shape is {}.'.format(float(loss)/ 10000))
print(' ')

#transfer pred flow data to original data lim

pred_ori = paddle.add(paddle.multiply(pred, paddle.subtract(max_values, min_values)), min_values)
print("---------------------------------------------------------------------------------------")

diff = paddle.subtract(flow , pred_ori)


plt.figure()
plt.subplot(321)
plt.scatter(X,Y,c = flow[:,0].numpy(),cmap=plt.cm.coolwarm,vmin = min(flow[:,0]),vmax = max(flow[:,0]))
#(50,50,r'Nancy',{'color':'r','fontsize':20})
plt.axis('square')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.colorbar()
plt.subplot(322)
plt.scatter(X,Y,c = pred_ori[:,0].numpy(),cmap=plt.cm.coolwarm,vmin = min(flow[:,0]),vmax = max(flow[:,0]))
#(50,50,r'Nancy',{'color':'r','fontsize':20})
plt.axis('square')
plt.colorbar()



plt.subplot(323)
plt.scatter(X,Y,c = flow[:,1],cmap=plt.cm.coolwarm,vmin = min(flow[:,1]),vmax = max(flow[:,1]))
#plt.text(60,60,r'Nancy',{'color':'r','fontsize':20})
plt.axis('square')
plt.colorbar()
plt.subplot(324)
plt.scatter(X,Y,c = pred_ori[:,1].numpy(),cmap=plt.cm.coolwarm,vmin = min(flow[:,1]),vmax = max(flow[:,1]))
#(50,50,r'Nancy',{'color':'r','fontsize':20})
plt.axis('square')
plt.colorbar()


plt.subplot(325)
plt.scatter(X,Y,c = flow[:,2],cmap=plt.cm.coolwarm,vmin = min(flow[:,2]),vmax = max(flow[:,2]))
#plt.text(50,50,r'Nancy',{'color':'r','fontsize':20})
#plt.axis(axis_limit)
plt.axis('square')
plt.colorbar()
plt.subplot(326)
plt.scatter(X,Y,c = pred_ori[:,2].numpy(),cmap=plt.cm.coolwarm,vmin = min(flow[:,2]),vmax = max(flow[:,2]))
#(50,50,r'Nancy',{'color':'r','fontsize':20})
plt.axis('square')
plt.colorbar()

plt.show()



###### compute drag
elements= pd.read_csv('./data/data/dataset_shapes_gcnn/cylindre/elements.csv').values
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
