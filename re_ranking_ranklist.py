import numpy as np
from scipy.spatial.distance import cdist

def re_ranking(probFea,galFea,rank,k1=18,k2=4,lambda_value=0.3):
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]    
    feat = np.append(probFea,galFea,axis = 0)
    feat = feat.astype(np.float16)
    original_dist = cdist(feat,feat).astype(np.float16)  
    original_dist = np.power(original_dist,2).astype(np.float16)
    del feat    
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float16)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])
    
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float16)

    
    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float16)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    index=[]
    dist=[]
    temp_dist = final_dist.copy()
    for k in range(rank):
        temp_index=np.argmin(temp_dist,1)
        index.append(temp_index)
        dist.append(np.min(temp_dist,1))
        for j in range(temp_dist.shape[0]):
            temp_dist[j][temp_index[j]]=np.inf
    return np.array(dist).astype('float32'),np.array(index)

    


