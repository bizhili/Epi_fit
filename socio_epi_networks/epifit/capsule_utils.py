import torch
import numpy as np
from collections import Counter
import copy
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

#sizes is number of age distribution,  capsule is a unit of household
def create_sparse_matrix_from_capsules(sizes, capsules, device=device):
    """
    Creates a sparse adjacency matrix for connections between people that are in the capsules/households.
    sizes (shape: number-of-age-groups) contains the size of each age group - each index corresponds
      to an age group contains the population size of that age group.
    capsules (shape: number-of-households) contains lists of the age groups of those within each
      household.
    The returned object (shape: size-of-population x size-of-population) contains at each index (i, j)
      whether person i has a connection with person j. The number of connections is given by the sum
      over all capsules of the number of combinations of two people within each capsule.
    Note that in the returned matrix, the people are ordered by age group. I.e. indices 0 to x correspond
      to individuals in the lowest age group and indices x+1 to y correspond to individuals in the next 
      largest group and so on.
    
    parameters: Tensor (1D, int), list (list of lists of ints)
    
    returns: sparse_coo_tensor (2D, float)
    """
    # Initiate indeces of row and column by the age group sizes
    cumsizes = [0] + sizes.cumsum(dim=0).tolist()
    n = cumsizes[-1]
    row_idx_cnt = Counter()
    for i in range(sizes.size()[0]):
        row_idx_cnt[i] = cumsizes[i]
    col_idx_cnt = copy.deepcopy(row_idx_cnt)

    # Building the indeces list
    indeces_l = list()
    for capsule in capsules:
        for person_1 in capsule:
            tmp_col_idx_cnt = copy.deepcopy(col_idx_cnt)
            for person_2 in capsule:
                if not row_idx_cnt[person_1] == col_idx_cnt[person_2]:  # Not adding self-encounters
                    indeces_l.append([row_idx_cnt[person_1], col_idx_cnt[person_2]])
                col_idx_cnt[person_2] += 1
            row_idx_cnt[person_1] += 1
            col_idx_cnt = copy.deepcopy(tmp_col_idx_cnt)
        col_idx_cnt = col_idx_cnt + Counter(capsule)

    # Creating sparse matrix of households
    i = torch.LongTensor(indeces_l)
    v = torch.FloatTensor([1 for i in range(len(indeces_l))])
    households_sparse_matrix = torch.sparse_coo_tensor(i.t(), v, torch.Size([n, n]))
    return households_sparse_matrix

#sizes is number of age distribution, create a adjacency matrix by meeting probability
def stochastic_block_model_sparse_undirected_triu(sizes, p, device=device):
    """
    Returns a sparse upper triangular adjacency matrix where edges of 1 mean a connection.
    sizes (shape: number-of-age-groups) contains the size of each age group - each index corresponds
      to an age group contains the population size of that age group.
    p (shape: number-of-age-groups x number-of-age-groups) contains at each index (i, j) the probability
      of a person in age group i meeting any one person in age group j.
    The returned object (shape: size-of-population x size-of-population) contains at each index (i, j)
      whether person i has a connection with person j.
    Note that sparsity depends on the probability of connections being relatively low.
      
    parameters: Tensor (1D), Tensor (2D, float), string
    
    returns: sparse_coo_tensor (2D, float)
    """
    sizes = sizes.int()
    con_idx = []
    nb = sizes.size()[0]
    cumsizes = [0] + sizes.cumsum(dim=0).tolist()
    n = cumsizes[-1]
    for i in range(nb):
        for j in range(i, nb):
            eij = (p[i, j] * sizes[i] * sizes[j]).int().item()
            src = [random.randint(cumsizes[i], cumsizes[i + 1] - 1) for _ in range(eij)]
            dst = [random.randint(cumsizes[j], cumsizes[j + 1] - 1) for _ in range(eij)]
            e = zip(src, dst)
            e = filter(lambda x: x[1] > x[0], e) # removes extra edges that happen when i==j
            con_idx += list(e)
    con_idx = torch.Tensor(con_idx).to(device).t()
    m = con_idx.size()[1]
    con_vals = torch.ones(m).to(device)
    con = torch.sparse_coo_tensor(con_idx, con_vals, size=(n, n))
    con = con.int().float()
    return con

# create a subset of sparse_matrix given indices at certain axis, 
# people are ordered by age group in this sparse matrix
def slice_sparse_matrix_by_indices(sparse_matrix, indices_to_slice, axis, device=device):
    """
    Creates sparse matrix only with the values of sparse_matrix at indices in indices_to_slice
    sparse_matrix (shape: size-of-population x size-of-population) is an adjacency matrix with
      connections represented by 1's. Note that in this matrix, the people are ordered by age group. 
    indices_to_slice (shape: size-of-indices-to-keep) contains indices to keep with values in sparse matrix
    axis denotes along which axis to filter indices
    
    parameters: sparse_coo_tensor (2D, float), list (int), int, string
    
    returns: sparse_coo_tensor (2D, float)
    """
    i = sparse_matrix.coalesce().indices()
    v = sparse_matrix.coalesce().values()

    mapping = torch.zeros(i.size()).byte().to(device)
    for idx in indices_to_slice:
        mapping = mapping | i[axis].eq(idx)

    # Getting the elements in `i` which correspond to `idx`:
    v_idx = mapping.byte()
    v_idx = v_idx.sum(dim=0).squeeze() == i.size(0)
    v_idx = v_idx.nonzero().squeeze()

    # Slicing `v` and `i` accordingly:
    v_sliced = v[v_idx]
    i_sliced = i.index_select(dim=1, index=v_idx)

    # To make sure to have a square dense representation:
    return torch.sparse.FloatTensor(i_sliced, v_sliced, sparse_matrix.size())

#from sparse matrix to probibility of meeting pairs, sizes is number of age group distribution
def reconstruct_block_model_from_sparse_matrix(sparse_matrix, sizes, device=device):
    """
    Creates matrix of contact probabilities between age groups if given a matrix with contacts
      between individuals.
    sparse_matrix (shape: size-of-population x size-of-population) is an adjacency matrix with
      connections represented by 1's. Note that in this matrix, the people are ordered by age group. 
      I.e. indices 0 to x correspond to individuals in the lowest age group and indices x+1 to y 
      correspond to individuals in the next largest group and so on.
    sizes (shape: number-of-age-groups) contains the size of each age group - each index corresponds
      to an age group contains the population size of that age group.
    The returned object (shape: number-of-age-groups x number-of-age-groups) contains at each index
      (i, j) the probability of contact between individuals in age groups or blocks i and j.
    
    parameters: sparse_coo_tensor (2D, float), Tensor (1D, int), string
    
    returns: Tensor (2D, float)
    """
    age_contact = []
    cumsizes = [0] + sizes.cumsum(dim=0).tolist()

    for i in range(sizes.size()[0]):
        age_group_encounters = slice_sparse_matrix_by_indices(sparse_matrix,
                                                              range(cumsizes[i], cumsizes[i + 1]),
                                                              1, device)
        age_group_contact = []
        for j in range(len(cumsizes) - 1):
            g_g_encounters = torch.sparse.sum(slice_sparse_matrix_by_indices(age_group_encounters,
                                                                             range(cumsizes[j], cumsizes[j + 1]),
                                                                             0, device))
            # Compute the contact rate per individual in block.
            #age_group_contact.append(g_g_encounters.item() / sizes[i].item())
            # Compute the contact probability per pair of individuals in blocks i j.
            age_group_contact.append(g_g_encounters.item() / sizes[i].item() / sizes[i].item())
        age_contact.append(age_group_contact)

    return torch.Tensor(age_contact).to(device).transpose(0,1)

#merge two adjacency matrix
def merge_contacts(X1, X2):
    Y = X1 + X2
    Y.coalesce()
    Y._values().fill_(1)
    return Y

#get the size of age group
# gets array of size population_size and returns aggregated array of size age_groups
def results_by_age_group(res, age_group):
    return torch.Tensor([res[age_group == g].sum() for g in age_groups])

#check if sparse matrix is symetrical
def check_if_sparse_matrix_is_symetrical(sparse_matrix):
    idx = sparse_matrix.coalesce().indices().cpu()
    return (np.sort(idx[0].numpy()) == np.sort(idx[1].numpy())).all()

#check if two sparse matrices are equal
def check_if_two_sparse_matrices_are_equal(sparse_1, sparse_2):
    idx_1 = sparse_1.coalesce().indices().cpu()
    idx_2 = sparse_2.coalesce().indices().cpu()
    is_equal_row_idx = (set(np.sort(idx_1[0].numpy())) == set(np.sort(idx_2[0].numpy())))
    is_equal_col_idx = (set(np.sort(idx_1[1].numpy())) == set(np.sort(idx_2[1].numpy())))
    return is_equal_row_idx and is_equal_col_idx



#######################################################################
## Tests
#######################################################################

import unittest

class Test_capsule_utils(unittest.TestCase):

    def test_dummy(self):
        pass

    def test_reconstruct_age_contact_from_sparse_matrix(self):
        n_test = 1800000
        sparse_test = torch.sparse_coo_tensor(
                      torch.randint(0,3000, size=(2,n_test)),
                      torch.ones(n_test),
                      size=(3000,3000))

        sparse_test = sparse_test.int().float()
        sizes_test = torch.IntTensor([1000]*3)
        # print(sizes_test.size()[0])

        result = reconstruct_block_model_from_sparse_matrix(sparse_test, sizes_test, device="cpu")

        for v in result.flatten():
            self.assertAlmostEqual(v.item(),200,-1)

        # expected contact rate of 200 in each cell


    def test_check_if_sparse_matrix_is_symetrical(self):

        con_idx = torch.Tensor(np.array([[0,1],[1,0]])).t()
        con_vals = torch.ones(con_idx.size()[1])
        n=2
        test_1 = torch.sparse_coo_tensor(con_idx,con_vals,size=(n,n)).int().float()
        result = check_if_sparse_matrix_is_symetrical(test_1)
        self.assertTrue(result)

        con_idx = torch.Tensor(np.array([[0,0],[1,0]])).t()
        con_vals = torch.ones(con_idx.size()[1])
        n=2
        test_1 = torch.sparse_coo_tensor(con_idx,con_vals,size=(n,n)).int().float()

        result = check_if_sparse_matrix_is_symetrical(test_1)
        self.assertFalse(result)

    def test_check_if_two_sparse_matrices_are_equal(self):
        con_idx = torch.Tensor(np.array([[0,1],[1,0]])).t()
        con_vals = torch.ones(con_idx.size()[1])
        n=2
        test_1 = torch.sparse_coo_tensor(con_idx,con_vals,size=(n,n)).int().float()
        test_2 = torch.sparse_coo_tensor(con_idx,con_vals,size=(n,n)).int().float()
        result = check_if_two_sparse_matrices_are_equal(test_1, test_2)
        self.assertTrue(result)

        con_idx = torch.Tensor(np.array([[0,1],[1,1]])).t()
        test_2 = torch.sparse_coo_tensor(con_idx,con_vals,size=(n,n)).int().float()
        result = check_if_two_sparse_matrices_are_equal(test_1, test_2)
        self.assertFalse(result)

    def test_stochastic_block_model_sparse_undirected_triu(self):
        sizes = torch.tensor([1000,1000],dtype=torch.int32)
        p = torch.Tensor([[100,20],[20,100]])
        G = stochastic_block_model_sparse_undirected_triu(sizes,p)
        pp = reconstruct_block_model_from_sparse_matrix(G,sizes,"cpu")
        pp = pp + pp.t()
        self.assertAlmostEqual(p[0, 0].item(), pp[0, 0].item(), 1)
        self.assertAlmostEqual(p[1, 0].item(), pp[1, 0].item(), 1)
        self.assertAlmostEqual(p[1, 1].item(), pp[1, 1].item(), 1)
        pass
if __name__=="__main__":
    unittest.main()
