import pandas as pd
import numpy as np
from statistics import mean
from epifit.capsule_utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"
    

def generate_households_contactsbyage(n, path_to_data="../", device=device):
    contact_rates_by_age = pd.read_csv(path_to_data+"data/contact_matrix.csv",header=0,index_col=0)
    age_dist = pd.read_csv(path_to_data+"data/geokg/age_structure_10yr.csv",header=0,index_col=0)
    household_type = pd.read_excel(path_to_data+"data/households/households_fixed.xlsx",sheet_name="household type")
    household_size = pd.read_excel(path_to_data+"data/households/households_fixed.xlsx",sheet_name="household size")
    household_ages = pd.read_excel(path_to_data+"data/households/households_fixed.xlsx",sheet_name="household ages")


    age_group_names = contact_rates_by_age.columns
    age_groups = range(len(list(contact_rates_by_age.columns)))
    age_dist = age_dist["total_prop"].to_numpy() 
    age_dist = torch.Tensor(age_dist).to(device)
    contact_rates_by_age = torch.Tensor(contact_rates_by_age.to_numpy()).to(device)


    # Calculating expectation of people amount in a household (= household_size)
    household_type_size = household_type.set_index('household type').join(household_size.set_index('household type'))

    household_size_np = household_type_size['avg household size'].to_numpy()
    type_prob_np = household_type_size['type prob'].to_numpy()
    type_size_prob_np = household_type_size['type size prob'].to_numpy()

    expectation_of_household_size = np.sum(household_size_np*type_prob_np*type_size_prob_np)
    
    # Calculating households amount
    households_amount = int(round(n/expectation_of_household_size))

    # Randomly choosing households
    households_random_list = list()
    age_groups_sizes_cnt = Counter()

    age_group_types = list(set(household_ages['age']))
    age_group_types.sort()
    household_ages_curr = household_ages.replace(age_group_types, list(age_groups))

    household_types_l = household_type['household type'].to_list()
    household_types_prob_l = household_type['type prob'].to_list()

    for i in range(households_amount):
        # choose randomly household_type
        rand_household_type = random.choices(household_types_l,
                                          weights=household_types_prob_l,
                                          k=1)[0]
        rand_household_type_size = household_size[household_size['household type'] == rand_household_type]

        # choose randomly household_size
        rand_household_size = eval(random.choices(rand_household_type_size['household size'].to_list(),
                                          weights=rand_household_type_size['type size prob'].to_list(),
                                          k=1)[0])

        rand_household_type_size_age = household_ages_curr.loc[(household_ages_curr['household type'] == rand_household_type) &
                                                      (household_ages_curr['avg household size'] == mean(rand_household_size))]

        # choose randomly ages inside the household   
        rand_household_ages = random.choices(rand_household_type_size_age['age'].to_list(),
                                          weights=rand_household_type_size_age['prob'].to_list(),
                                          k=random.choice(rand_household_size))

        age_groups_sizes_cnt = age_groups_sizes_cnt + Counter(rand_household_ages)
        households_random_list.append(rand_household_ages)

    age_sizes = torch.Tensor(np.asarray([age_groups_sizes_cnt[i] for i in range(len(age_groups_sizes_cnt))])).long().to(device)
    contacts_households = create_sparse_matrix_from_capsules(age_sizes, households_random_list).to(device)
    block_model_households = reconstruct_block_model_from_sparse_matrix(contacts_households, age_sizes, device)

    block_model_total = contact_rates_by_age / age_sizes

    block_model_outofhome = (block_model_total - block_model_households)

    # new population size
    n = sum(list(age_groups_sizes_cnt.values()))

    age_group = range(len(age_groups))
    age_group = torch.Tensor(age_group).to(device)
    age_group = torch.repeat_interleave(age_group,age_sizes).int()
    age_group = age_group.tolist()

    contacts_outofhome = stochastic_block_model_sparse_undirected_triu(age_sizes,block_model_outofhome).to(device) # a matix that take in counter the ages but not households
    contacts_outofhome = contacts_outofhome + contacts_outofhome.t()
    contacts_insteadofhouseholds = stochastic_block_model_sparse_undirected_triu(age_sizes,block_model_households).to(device)
    contacts_insteadofhouseholds = contacts_insteadofhouseholds + contacts_insteadofhouseholds.t()

    # randomly assign ages within age groups - note this relies on the specific age groupings in data from the spreadsheets above
    ages = torch.Tensor([random.randint(i*10, (i+1)*10-1) for i in age_group]).to(device)

    return contacts_outofhome, contacts_insteadofhouseholds, ages


def generate_social_layer_from_physical(contacts_insteadofhouseholds, contacts_outofhome, ages, age_group_boundaries, connection_by_age_group, percent_physical_to_social_connection,device=device):
    # connection_by_age_group is a list of how likely a person in each age group is to be socially connected to another in the same age group
    # percent_physical_to_social_connection is how likely people who meet in the physical world are to be connected on social media
    
    con_socio = contacts_insteadofhouseholds # those in families have social influence on each other
    
    num_age_groups = age_group_boundaries.size()[0]
    n = ages.size()[0]
    age_sizes = torch.Tensor([torch.nonzero(ages<=b)[-1]+1 for b in age_group_boundaries]).to(device)
    age_sizes[1:] = age_sizes[1:] - age_sizes[:-1]
    
    block_model_age_contacts = torch.eye(num_age_groups).to(device)*connection_by_age_group
    age_connections = stochastic_block_model_sparse_undirected_triu(age_sizes, block_model_age_contacts) # people in same age group are also more likely to be connected on social media
    con_socio += age_connections + age_connections.t()
    
    physical_indices = contacts_outofhome.coalesce().indices()
    indices_to_keep = random.sample(range(physical_indices.size()[-1]), int(physical_indices.size()[-1]*percent_physical_to_social_connection))
    indices = physical_indices[:, indices_to_keep]
    physical_connections = torch.sparse.FloatTensor(indices=indices, values=torch.ones(indices[0].size()[0]).to(device), size=(n, n))
    con_socio += physical_connections + physical_connections.t()
    
    return con_socio

def generate_random_network(n, degree, device=device):
    m = int(degree*n / 2) # number of connections in contact matrix
    idx = torch.randint(n,(2,m)).to(device)
    con = torch.sparse.FloatTensor(idx,torch.ones(m).to(device),torch.Size([n,n])).to(device)
    return con + con.t()

    
