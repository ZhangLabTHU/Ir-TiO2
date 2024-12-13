# copyright THU-Zhanglab
# coding:utf-8 
import random
import numpy as np
from ase import units
from ase.io import read
import pickle
from random import sample
from scipy.stats import norm
from multiprocessing import Pool
import os
import warnings
warnings.filterwarnings("ignore")


p = read('pure.traj')
# neighboring atoms of 5-coordinated surface site
Top_indice = [167,175,183,191,159,151,143,135,]
top_site_neighbor_shell = {}
for index in Top_indice:
    distances = p.get_distances(index, [i for i in range(len(p))], mic=True, vector=False)
    ir_ti_distance_dict = {}
    for i in range(len(p)):
        if p[i].symbol != 'O':
            ir_ti_distance_dict[i] = distances[i]      
    sorted_ir_ti_distance_dict = sorted(ir_ti_distance_dict.items(), key=lambda d: d[1], reverse=False)
    shell0 = [index]
    shell1 = [sorted_ir_ti_distance_dict[1][0],sorted_ir_ti_distance_dict[2][0]]
    shell2 = [sorted_ir_ti_distance_dict[3][0],sorted_ir_ti_distance_dict[4][0]]
    shell3 = [sorted_ir_ti_distance_dict[5][0],sorted_ir_ti_distance_dict[6][0],
              sorted_ir_ti_distance_dict[7][0],sorted_ir_ti_distance_dict[8][0]]
    shell4 = [sorted_ir_ti_distance_dict[9][0],sorted_ir_ti_distance_dict[10][0]]
    top_site_neighbor_shell[index] = shell0 + shell1 + shell2 + shell3 + shell4

# neighboring atoms of 6-coordinated surface site
Top_indice2  = [134,142,150,158,166,174,182,190]
for index in Top_indice2:
    distances = p.get_distances(index, [i for i in range(len(p))], mic=True, vector=False)
    ir_ti_distance_dict = {}
    for i in range(len(p)):
        if p[i].symbol != 'O':
            ir_ti_distance_dict[i] = distances[i]      
    sorted_ir_ti_distance_dict = sorted(ir_ti_distance_dict.items(), key=lambda d: d[1], reverse=False)
    shell0 = [index]
    shell1 = [sorted_ir_ti_distance_dict[1][0],sorted_ir_ti_distance_dict[2][0]]
    shell2 = [sorted_ir_ti_distance_dict[7][0],sorted_ir_ti_distance_dict[8][0]]
    shell3 = [sorted_ir_ti_distance_dict[3][0],sorted_ir_ti_distance_dict[4][0],
              sorted_ir_ti_distance_dict[5][0],sorted_ir_ti_distance_dict[6][0]]
    shell4 = [sorted_ir_ti_distance_dict[9][0],sorted_ir_ti_distance_dict[10][0]]
    top_site_neighbor_shell[index] = shell0 + shell1 + shell2 + shell3 + shell4


def get_feature_from_config(config_int, top_site_neighbor_shell):
    Feature = []
    doping_atom_list = [i for i,x in enumerate(config_int) if x==1]
    ratio = len(doping_atom_list)/64

    for top_index in Top_indice:
        site = int(config_int[top_index])

        adsorption = []
        shell = top_site_neighbor_shell[top_index]
        shell_symbol = [1 if i in doping_atom_list else 0 for i in shell]

        adsorption = []
        adsorption.append([shell_symbol[0]].count(0))
        adsorption.append([shell_symbol[0]].count(1))
        adsorption.append(shell_symbol[1:3].count(0))
        adsorption.append(shell_symbol[1:3].count(1))
        
        count_shell1_ti_neighbor_shell1_ti_num = 0
        count_shell1_ir_neighbor_shell1_ir_num = 0
        count_shell1_ti_neighbor_shell3_ti_num = 0
        count_shell1_ir_neighbor_shell3_ir_num = 0
        for i in shell[1:3]:
            shell_i_neieghbor_shell1 = top_site_neighbor_shell[i][1:3]
            shell_i_neieghbor_shell1_symbol = [1 if j in doping_atom_list else 0 for j in shell_i_neieghbor_shell1]
            shell_i_neieghbor_shell3 = top_site_neighbor_shell[i][5:9]
            shell_i_neieghbor_shell3_symbol = [1 if j in doping_atom_list else 0 for j in shell_i_neieghbor_shell3]
            if i in doping_atom_list: # i is ir
                count_shell1_ir_neighbor_shell1_ir_num += shell_i_neieghbor_shell1_symbol.count(1)
                count_shell1_ir_neighbor_shell3_ir_num += shell_i_neieghbor_shell3_symbol.count(1) 
            else:
                count_shell1_ti_neighbor_shell1_ti_num += shell_i_neieghbor_shell1_symbol.count(0)
                count_shell1_ti_neighbor_shell3_ti_num += shell_i_neieghbor_shell3_symbol.count(0) 
        adsorption.append(count_shell1_ti_neighbor_shell1_ti_num)
        adsorption.append(count_shell1_ir_neighbor_shell1_ir_num)
        adsorption.append(count_shell1_ti_neighbor_shell3_ti_num)
        adsorption.append(count_shell1_ir_neighbor_shell3_ir_num)
        
        adsorption.append(shell_symbol[3:5].count(0))
        adsorption.append(shell_symbol[3:5].count(1))
        adsorption.append(shell_symbol[5:9].count(0))
        adsorption.append(shell_symbol[5:9].count(1))
        
        # add angle information if there are 2 Ir atoms in 4 atoms in 4th shell
        shell_4 = shell[5:9]
        shell_4_symbol = shell_symbol[5:9]  
        angle = 0
        if site == 0 and shell_4_symbol.count(1) == 2: # Ti reaction site, 2 Ir atoms in the forth shell
            indices_2_ir = np.where(np.array(shell_4_symbol)==1) 
            index_ir_1 = shell_4[indices_2_ir[0][0]]
            index_ir_2 = shell_4[indices_2_ir[0][1]]
            angle = p.get_angle(index_ir_1,top_index,index_ir_2,mic=True)
        
        if site == 0 and shell_4_symbol.count(1) in [3,4]: # Ti reaction site, 3 or 4 Ir atoms in the forth shell
            angle = 48
        
        if site == 1 and shell_4_symbol.count(0) == 2: # Ir reaction site, 2 Ti atoms in the forth shell
            indices_2_ti = np.where(np.array(shell_4_symbol)==0) 
            index_ti_1 = shell_4[indices_2_ti[0][0]]
            index_ti_2 = shell_4[indices_2_ti[0][1]]
            angle = p.get_angle(index_ti_1,top_index,index_ti_2,mic=True)
        if site == 1 and shell_4_symbol.count(0) in [3,4]: # Ti reaction site, 3 or 4 Ir atoms in the forth shell
            angle = 48
        adsorption.append(int(angle))

        count_shell3_ti_neighbor_shell1_ti_num = 0
        count_shell3_ir_neighbor_shell1_ir_num = 0
        count_shell3_ti_neighbor_shell3_ti_num = 0
        count_shell3_ir_neighbor_shell3_ir_num = 0
        
        for i in shell[5:9]:
            shell_i_neieghbor_shell1 = top_site_neighbor_shell[i][1:3]
            shell_i_neieghbor_shell1_symbol = [1 if j in doping_atom_list else 0 for j in shell_i_neieghbor_shell1]
            shell_i_neieghbor_shell3 = top_site_neighbor_shell[i][5:9]
            shell_i_neieghbor_shell3_symbol = [1 if j in doping_atom_list else 0 for j in shell_i_neieghbor_shell3]
            if i in doping_atom_list: # i is ir
                count_shell3_ir_neighbor_shell1_ir_num += shell_i_neieghbor_shell1_symbol.count(1)
                count_shell3_ir_neighbor_shell3_ir_num += shell_i_neieghbor_shell3_symbol.count(1) 
            else:
                count_shell3_ti_neighbor_shell1_ti_num += shell_i_neieghbor_shell1_symbol.count(0)
                count_shell3_ti_neighbor_shell3_ti_num += shell_i_neieghbor_shell3_symbol.count(0) 
        adsorption.append(count_shell3_ti_neighbor_shell1_ti_num)
        adsorption.append(count_shell3_ir_neighbor_shell1_ir_num)
        adsorption.append(count_shell3_ti_neighbor_shell3_ti_num)
        adsorption.append(count_shell3_ir_neighbor_shell3_ir_num)
        
        adsorption.append(shell_symbol[9:].count(0))
        adsorption.append(shell_symbol[9:].count(1))
        Feature.append(adsorption + [ratio])
    return Feature

class MenteCarlo:
    def __init__(self, parameter):
        self.parameter = parameter
        self.temperature = parameter[0]
        self.n_step = parameter[1]
        self.out_path = parameter[2]
        self.max_pre_gen = parameter[3]
        self.length = parameter[4] 
        self.initial_config = read('pure.traj')
        self.avail_m2_indices = [i for i in range(len(self.initial_config)) 
                                if self.initial_config[i].tag == 1]
        self.preprocessing = pickle.load(open("Preprocessing.pkl", "rb"))
        self.model = pickle.load(open("GPRmodel.model", "rb"))
        
        # initilize config by random generate
        m2_list = sample(self.avail_m2_indices, self.length) # initialise popluation
        self.config = [1 if i in m2_list else 0 for i in range(len(self.initial_config))]
        print('Initilize...')
    
    def evaluate(self, config_int, epsilon =0.01):
        feature = get_feature_from_config(config_int = config_int, top_site_neighbor_shell = top_site_neighbor_shell)
        feature = self.preprocessing.transform(feature)
        go_goh_energies, go_goh_energies_sigma = self.model.predict(feature, return_std=True)
        current = [8.0636*go_goh -12.9764 if go_goh < 1.53 
                            else -8.7356*go_goh + 12.7264 for go_goh in go_goh_energies]
        current_sigma = [8.0636*go_goh_sigma if go_goh < 1.53 else 8.7356*go_goh_sigma
                        for (go_goh, go_goh_sigma) in zip(go_goh_energies, go_goh_energies_sigma)]
        u = np.mean(current)
        sigma = (np.mean([i**2 for i in current_sigma])) ** 0.5
        Z = (u - self.max_pre_gen)/sigma
        EI = (u - self.max_pre_gen - epsilon) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -EI, u, sigma
    
    def get_mutated_config_int(self, config_int=[]):
        mutated_config_int = config_int.copy()

        # choose two index to be exchanged
        change_index_1 = random.choice(self.avail_m2_indices)

        # available indices to be exchanged should have different element
        avail_change_indices = []
        for i in range(len(config_int)):
            if config_int[change_index_1] != config_int[i]:
                avail_change_indices.append(i)
        change_index_2 = random.choice(avail_change_indices)
        
        # make exchange
        mutated_config_int[change_index_2] = config_int[change_index_1]
        mutated_config_int[change_index_1] = config_int[change_index_2]
        
        return mutated_config_int
    
    def perform_mc_iterations(self,):
        print('Start MC interation:')
        # current ei and it's config int
        EI = []
        
        # global minimum ei and it's config int, activity and sigma
        EI_min = []
        Activity_ei_min = []
        Sigma_ei_min = []
        Config_ei_min = []
        
        # initialize 
        config_int = self.config.copy()
        ei_prior, activity_prior, sigma_prior  = self.evaluate(config_int)
        ei_min = ei_prior
        activity_ei_min = activity_prior
        sigma_ei_min = sigma_prior
        config_int_ei_min =  config_int

        for step in range(self.n_step):
            config_int_mutated = self.get_mutated_config_int(config_int)
            ei_mutated, activity_mutated, sigma_mutated = self.evaluate(config_int_mutated)
            delta_ei = ei_mutated - ei_prior
            theta = np.random.rand()
            mi = - delta_ei / (units.kB * self.temperature)
            mi = np.float(mi)
            p_a = np.exp(mi)
            prefilter = p_a - theta
            
            # accept
            if delta_ei < 0:
                ei_prior = ei_mutated
                config_int = config_int_mutated

                # update energy_max and config_int_max
                if ei_mutated < ei_min:
                    ei_min = ei_mutated
                    activity_ei_min = activity_mutated
                    sigma_ei_min = sigma_mutated
                    config_int_ei_min = config_int_mutated
            
            # accept with a probability 
            elif prefilter > 0:
                ei_prior = ei_mutated
                config_int = config_int_mutated                
            # no action
            else:
                pass
            
            # store config and ei per 100 steps
            if step % 10 == 0:
                EI.append(ei_prior)

                EI_min.append(ei_min)
                Activity_ei_min.append(activity_ei_min)
                Sigma_ei_min.append(sigma_ei_min)
                Config_ei_min.append(config_int_ei_min)
            
            # print current step per 500 steps
            if step % 10 == 0:
                print("############### Generation {} ###############".format(step))
        
        # make dictionary
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)        
        np.savetxt(self.out_path + '/ei_current.csv', EI, delimiter=',')
        np.savetxt(self.out_path + '/ei_min.csv', EI_min, delimiter=',')
        np.savetxt(self.out_path + '/activity_ei_min.csv', Activity_ei_min, delimiter=',')
        np.savetxt(self.out_path + '/sigma_ei_min.csv', Sigma_ei_min, delimiter=',')
        np.savetxt(self.out_path + '/config_ei_min.csv', Config_ei_min, delimiter=',')
        print('End!')


def mc_run(length):
    temperature = 100
    n_step = 300
    length = length
    out_path = str(length)
    max_last_gen = 0.13
    parameter = [temperature, n_step, out_path, max_last_gen, length]
    mc = MenteCarlo(parameter)
    mc.perform_mc_iterations()

from multiprocessing import Pool
if __name__ == "__main__":
    Length = [i for i in range(1,16)]
    pool = Pool(2)
    pool.map(mc_run, Length)
    pool.close()
    pool.join()