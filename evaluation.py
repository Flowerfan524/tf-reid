import numpy as np
from scipy.spatial.distance import cdist

def compute_ap(good_idx, junk_idx, pred_idx):
    cmc = np.zeros(pred_idx.shape)
    ngood = good_idx.shape[0]
    old_recall = 0.0
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for i,idx in enumerate(pred_idx):
        flag = 0
        if idx in good_idx:
            cmc[i-njunk:] = 1
            flag = 1
            good_now += 1
        elif idx in junk_idx:
            njunk += 1
            continue
        if flag == 1:
            intersect_size += 1

        recall = intersect_size/ngood
        precision = intersect_size/(j+1)
        ap = ap + (recall - old_recall)*((old_precision+precision)/2)
        old_recall = recall
        old_precision = precision
        j += 1
        if good_now == ngood:
            return ap,cmc
    if good_now < good:
        raise 'something wrong'

def evaluate(cos_dist,y1,y2,cam1,cam2):
    cmc = np.zeros(shape=cos_dist.shape)
    ap = np.zeros(cos_dist.shape[0])
    for idx,cls in enumerate(y1):
        print('processing {}/{} query file'.format(idx+1, cos_dist.shape[0]))
        good_idx = np.intersect1d(np.where(y2 == cls)[0],
                np.where(cam2 != cam1[idx])[0])
        junk_idx1 = np.intersect1d(np.where(y2 == cls)[0],
                np.where(cam2 == cam1[idx])[0])
        junk_idx2 = np.where(y2 == -1)[0]
        junk_idx = np.union1d(junk_idx1, junk_idx2)
        pred_idx = np.argsort(cos_dist[idx,:])
        ap[idx],cmc[idx,:] = compute_ap(good_idx, junk_idx, pred_idx)
    fcmc = np.mean(cmc, axis = 0)
    print('map:{}, r1_precision:{}'.format(np.mean(ap),fcmc[0]))
    return None

if __name__ == '__main__':
    query_file = '/tmp/Market-1501/feature/query.npz'
    test_file = '/tmp/Market-1501/feature/test.npz'
    query_data = np.load(query_file)
    test_data = np.load(test_file)

    query_feature = query_data['feature']
    test_feature = test_data['feature']
    #query_sum = np.sum(query_data['feature']**2,axis=0)
    #query_feature = query_data['feature']**2/query_sum[np.newaxis,:]
    #test_sum = np.sum(test_data['feature']**2,axis=0)
    #test_feature = test_data['feature']**2/test_sum[np.newaxis,:]
    cos_dist = cdist(query_feature, test_feature, metric='cosine')

    cmc = np.zeros(shape=cos_dist.shape)
    ap = np.zeros(cos_dist.shape[0])

    #for idx,cls in enumerate(query_data['label']):
    #    print('processing {}/{} query file'.format(idx+1, cos_dist.shape[0]))
    #    good_idx = np.intersect1d(np.where(test_data['label'] == cls)[0],
    #            np.where(test_data['cam'] != query_data['cam'][idx])[0])
    #    junk_idx1 = np.intersect1d(np.where(test_data['label'] == cls)[0],
    #            np.where(test_data['cam'] == query_data['cam'][idx])[0])
    #    junk_idx2 = np.where(test_data['label'] == -1)[0]
    #    junk_idx = np.union1d(junk_idx1, junk_idx2)
    #    pred_idx = np.argsort(cos_dist[idx,:])
    #    ap[idx],cmc[idx,:] = compute_ap(good_idx, junk_idx, pred_idx)

    #fcmc = np.mean(cmc, axis = 0)
    #print('map:{}, r1_precision:{}'.format(np.mean(ap),fcmc[0]))
    
    evaluate(cos_dist,query_data['label'],test_data['label'],query_data['cam'],test_data['cam'])
