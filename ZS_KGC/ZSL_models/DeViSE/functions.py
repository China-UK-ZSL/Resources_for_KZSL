import numpy as np
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
    file_path = os.path.join(data_path, 'semantic_embeddings')
    if dataset == 'NELL':
        if type == 'rdfs':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_55000.npz')
        elif type == 'rdfs_hie':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_hie_60000.npz')
        elif type == 'rdfs_cons':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_cons_60000.npz')
        elif type == 'text':
            file_name = os.path.join(file_path, 'rela_matrix_text.npz')
        elif type == 'rdfs_text':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_55000_text140.npz')
        else:
            print("WARNING: invalid semantic embeddings type")
    elif dataset == 'Wiki':
        if type == 'rdfs':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_65000.npz')
        elif type == 'rdfs_hie':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_hie_60000.npz')
        elif type == 'rdfs_cons':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_cons_60000.npz')
        elif type == 'text':
            file_name = os.path.join(file_path, 'rela_matrix_text.npz')
        elif type == 'rdfs_text':
            file_name = os.path.join(file_path, 'rela_matrix_rdfs_65000_text140.npz')
        else:
            print("WARNING: invalid semantic embeddings type")





    if file_name:
        rela_embeddings = np.load(file_name)['relaM'].astype('float32')


    else:
        print('WARNING: invalid semantic embeddings file path')
    return rela_embeddings

