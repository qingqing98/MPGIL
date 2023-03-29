

import pandas as pd
import numpy as np
import tensorflow as tf
from spektral.layers import GCNConv,GATConv
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dropout,concatenate,Dense,Dropout
from tensorflow.keras.initializers import GlorotUniform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model 
import scipy.io
from sklearn import metrics

cluster=num #Num is the number of cancer subtype recognition clusters. In AML cancer, num=4.
ff = 0
foldss = scipy.io.loadmat('.../data/AML_folds.mat')
folds = foldss['folds']
we = folds[0,ff]
we=we.astype(np.float32)


data1=pd.read_csv(".../data/extended_data/exp_e.CSV",index_col=0);
data2=pd.read_csv(".../data/extended_data/meth_e.CSV",index_col=0);
data3=pd.read_csv(".../data/extended_data/mirna_e.CSV",index_col=0);

data1=data1.T
data2=data2.T
data3=data3.T

data1_affinity=pd.read_csv(".../data/extended_data/exphighadj_e.CSV",index_col=0);
data2_affinity=pd.read_csv(".../data/extended_data/methhighadj_e.CSV",index_col=0);
data3_affinity=pd.read_csv(".../data/extended_data/mirnahighadj_e.CSV",index_col=0);

adj1=data1_affinity
adj2=data2_affinity
adj3=data3_affinity

nsample1 = data1.shape[0]
nsample2 = data2.shape[0]
nsample3 = data3.shape[0]

nfeature1 = data1.shape[1]
nfeature2 = data2.shape[1]
nfeature3 = data3.shape[1]

print(data1.shape)
print(data2.shape)
print(data3.shape)

def Network():
    initializer = GlorotUniform(seed=7)
    #==============================omics1======================================
    X1_in = Input(shape=(nfeature1,),name='omics1')
    A1_in = Input(shape=(nsample1,),sparse=False,name='adj1')
    h1 = GATConv(channels=512,  kernel_initializer=initializer, activation="relu")([X1_in,A1_in])
    h11 = Dropout(0.2)(h1)
    encoder_layer1=GATConv(channels=200, kernel_initializer=initializer)([h11,A1_in])
    #==============================omics2======================================
    X2_in = Input(shape=(nfeature2,),name='omics2')
    A2_in = Input(shape=nsample2,name='adj2')
    h2 = GATConv(channels=512,  kernel_initializer=initializer, activation="relu")([X2_in,A2_in])
    h22 =Dropout(0.2)(h2)
    encoder_layer2=GATConv(channels=200, kernel_initializer=initializer)([ h22,A2_in])
    #==============================omics3======================================
    X3_in = Input(shape=nfeature3,name='omics3')
    A3_in = Input(shape=nsample3,name='adj3')
    h3 = GATConv(channels=512,  kernel_initializer=initializer, activation="relu")([X3_in,A3_in])
    h33 =Dropout(0.2)(h3)
    encoder_layer3=GATConv(channels=200, kernel_initializer=initializer)([ h33,A3_in])
    #======================================The weight fusion layer=================================
    summ = tf.matmul(tf.linalg.diag(we[:,0]), encoder_layer1)+tf.matmul(tf.linalg.diag(we[:,1]), encoder_layer2)+tf.matmul(tf.linalg.diag(we[:,2]),encoder_layer3) 
    wei = 1/tf.reduce_sum(we,1)
    z_mean=tf.linalg.matmul(tf.linalg.diag(wei), summ)
    
    #=========================================A decoder======================================
    adj_dim=[128,256,512]
    hda = tf.keras.layers.Dense(units=adj_dim[0], activation=tf.nn.relu)(z_mean)
    hda1 = tf.keras.layers.Dense(units=adj_dim[1], activation=tf.nn.relu)(hda)
    decodeA_layer1=tf.keras.layers.Dense(units=nsample1, activation=tf.nn.softplus,
                                              name='reconstructionA1')(hda1)
    decodeA_layer2=tf.keras.layers.Dense(units=nsample2, activation=tf.nn.softplus,
                                              name='reconstructionA2')(hda1)
    decodeA_layer3=tf.keras.layers.Dense(units=nsample3, activation=tf.nn.softplus,
                                              name='reconstructionA3')(hda1)

    #=========================================X Decoder========================================

    dec_dim=[128,256,512]

    hdx = tf.keras.layers.Dense(units=dec_dim[0], activation=tf.nn.relu)(z_mean)
    hdx1 = tf.keras.layers.Dense(units=dec_dim[1], activation=tf.nn.relu)(hdx)
    decodeX_layer1=tf.keras.layers.Dense(units=nfeature1, activation=tf.nn.softplus,
                                              name='reconstructionX1'
                                            )(hdx1)  
    decodeX_layer2=tf.keras.layers.Dense(units=nfeature2, activation=tf.nn.softplus,
                                              name='reconstructionX2')(hdx1)
    decodeX_layer3=tf.keras.layers.Dense(units=nfeature3, activation=tf.nn.softplus,
                                              name='reconstructionX3')(hdx1)
    
     # classifier branch
    n_cluster=cluster
    classifier_layer = tf.keras.layers.Dense(n_cluster, \
                                                   activation=tf.nn.softmax,
                                                   name='classification')(z_mean) 
        
    return Model(inputs=[X1_in,X2_in,X3_in,A1_in,A2_in,A3_in],
                 outputs=[classifier_layer,z_mean,decodeX_layer1,decodeX_layer2,decodeX_layer3,
                          decodeA_layer1,decodeA_layer2,decodeA_layer3])
          
model=Network()
model.summary()

classifier_weight=0.2
X_weight=0.4
A_weight=0.4

model.compile(loss={'classification': 'categorical_crossentropy',
                         'reconstructionX1': 'mean_squared_error',
                         'reconstructionX2': 'mean_squared_error',
                         'reconstructionX3': 'mean_squared_error',
                         'reconstructionA1': 'mean_squared_error','reconstructionA2': 'mean_squared_error',
                         'reconstructionA3':'mean_squared_error'}, \
                   loss_weights={'classification': classifier_weight,
                                 'reconstructionX1': X_weight,'reconstructionX2': X_weight,
                                 'reconstructionX3':X_weight,
                                 'reconstructionA2':A_weight,
                                 'reconstructionA3':A_weight,
                                 'reconstructionA1':A_weight
                                 },
                   optimizer=tf.keras.optimizers.Adam())    

def cluster(data1,data2,data3,n_cluster):
        n = np.min((data1.shape[0],data2.shape[0],data3.shape[0],data1.shape[1],data2.shape[1],
                    data3.shape[1]))
        pca = PCA(n_components=n)
        pcs = pca.fit_transform(np.hstack((data1,data2,data3)))
        var = (pca.explained_variance_ratio_).cumsum()
        npc_raw = (np.where(var > 0.7))[0].min()  # number of PC used in K-means
        pcs = pcs[:, :npc_raw]
        # K-means clustering on PCs
        kmeans = KMeans(n_clusters=n_cluster, random_state=1).fit(StandardScaler().fit_transform(pcs))
        clustering_label = kmeans.labels_
        dummy_label = to_categorical(clustering_label)
        return dummy_label

dummy_label1=cluster(data1,data2,data3,n_cluster=cluster)

his = model.fit([data1,data2,data3,adj1,adj2,adj3],
                          {'classification': dummy_label1,
                            'reconstructionX1': data1,'reconstructionX2': data2,'reconstructionX3': data3,
                            'reconstructionA1': adj1,'reconstructionA2': adj2,'reconstructionA3': adj3},
                          batch_size=197,
                          epochs=150,
                           shuffle=True)

output=model.predict([data1,data2,data3,adj1,adj2,adj3],batch_size=197)
classes=output[0]
Z=output[1]
 
labels = KMeans(n_clusters=cluster, n_init=150).fit_predict(Z)
labels1=pd.DataFrame(labels)

out_file = '.../labels.CSV'
labels1.to_csv(out_file, header=True, index=True, sep=',')

sli=metrics.silhouette_score(Z,labels)
ch=metrics.calinski_harabasz_score(Z, labels)
db=metrics.davies_bouldin_score(Z,labels)



