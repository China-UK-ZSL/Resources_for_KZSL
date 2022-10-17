import scipy.io as scio
import os

def load_semantic_embed(data_path, dataset, type):
    """
    Load Semantic Embedding file.

    Parameters
    ----------
    file_name : str
        Name of the semantic embedding file.
    type: str
        Type of semantic embeddings, including

    Returns
    -------
    embeddings : NumPy arrays
       the size is
    Examples
    --------
    """

    file_name = ''

    if dataset == 'AwA2':
        file_path = os.path.join(data_path, 'semantic_embeddings')
        if type == 'att':
            file_name = os.path.join(data_path, 'binaryAtt_splits.mat')
        elif type == 'w2v':
            file_name = os.path.join(file_path, 'awa_w2v.mat')
        elif type == 'w2v-glove':
            file_name = os.path.join(file_path, 'awa_w2v_glove.mat')
        elif type == 'hie':
            file_name = os.path.join(file_path, 'awa_hierarchy_gae.mat')
        elif type == 'kge':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000.mat')
        elif type == 'kge_text':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000_text_140.mat')
        elif type == 'kge_facts':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_Facts_60000_80000.mat')
        elif type == 'kge_logics':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_Logics_70000.mat')
        else:
            print("WARNING: invalid semantic embeddings type")

    else:
        file_path = os.path.join(data_path, dataset, 'semantic_embeddings')
        if type == 'hie':
            file_name = os.path.join(file_path, 'hierarchy_gae.mat')
        elif type == 'w2v':
            file_name = os.path.join(data_path, 'w2v.mat')
        elif type == 'w2v-glove':
            file_name = os.path.join(file_path, 'w2v_glove.mat')
        elif type == 'att':
            file_name = os.path.join(file_path, 'atts_binary.mat')
        # else:
        #     print('WARNING: invalid semantic embeddings type')



        if dataset == 'ImNet_A':
            if type == 'kge':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000.mat')
            elif type == 'kge_text':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000_text_nei_140.mat')
            elif type == 'kge_facts':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_Facts_60000_70000.mat')
        if dataset == 'ImNet_O':
            if type == 'kge':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000.mat')
            elif type == 'kge_text':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000_text_nei_140.mat')
            elif type == 'kge_facts':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_Facts_60000_70000.mat')


    if file_name:
        matcontent = scio.loadmat(file_name)
        if dataset == 'AwA2':
            if type == 'att':
                cls_embeddings = matcontent['att'].T
            else:
                cls_embeddings = matcontent['embeddings']
        else:
            if type == 'w2v':
                cls_embeddings = matcontent['w2v'][:2549]
            else:
                cls_embeddings = matcontent['embeddings']
    else:
        print('WARNING: invalid semantic embeddings file path')
    return cls_embeddings

