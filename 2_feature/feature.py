from operator import ne
import ase.db
from ase.io import read
import numpy as np
from ase import Atom,Atoms
from ase.neighborlist import NeighborList
from mendeleev import *

p = read('pure.traj')
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

Site = []
E_o = []
E_oh = []
Feature = []
Energy = []
Index_Site_Energy = []
db = ase.db.connect('../1_database/tio2_ir_o_oh.db')
for row in db.select(jobtype='ads_o', status='relaxed'):
    print(row.id)
    sid = row.sid
    uid = row.uid
    begin_index = sid.find('0x')
    config_code = sid[begin_index:-3]
    config_str = '{:0192b}'.format(int(config_code,16))
    config_int = [int(i) for i in config_str]
    doping_atom_list = [i for i,x in enumerate(config_int) if x==1]
    ratio = len(doping_atom_list)/64

    top_index = int(uid[-3:])
    site = int(config_int[top_index])
    Site.append(site)

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

    sur_energy = db.get(sid=row.sid, jobtype='sur').energy
    ads_energy_o = row.energy
    ads_energy_oh = db.get(uid=row.uid, jobtype='ads_oh').energy 
    d_eo = ads_energy_o - sur_energy + 7.46
    d_eoh = ads_energy_oh - sur_energy + 10.84
    E_o.append(d_eo)
    E_oh.append(d_eoh)
    d_go = d_eo + 0.05
    d_goh = d_eoh + 0.35
    Energy.append(d_go - d_goh)
    print(site, d_go-d_goh)
    Index_Site_Energy.append([row.id, ratio,site, d_go-d_goh])
    go_goh =  d_go - d_goh

np.savetxt('feature.csv', Feature, delimiter=',')
np.savetxt('energy.csv', Energy, delimiter=',')
np.savetxt('energy_o.csv', E_o, delimiter=',')
np.savetxt('energy_oh.csv', E_oh, delimiter=',')
np.savetxt('index_site_energy.csv', Index_Site_Energy, delimiter=',')
