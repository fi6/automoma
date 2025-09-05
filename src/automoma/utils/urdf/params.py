import sys

import numpy as np
import trimesh


def get_params(mobility_id):

    sphere_params = {}
    scale_params = {}
    prior_tip_link = ""

    if mobility_id == "12561":
        sphere_params = {"link_0": 4, "link_1": 3} 
        scale_params = {"link_1": 0.9} 

    elif mobility_id == "46197":
        sphere_params = {"link_0": 7, "link_1": 15}  
        scale_params = {"link_0": 0.9} 

    elif mobility_id == "101773":
        sphere_params = {"link_0": 1, "link_9": 5}  
        scale_params = {"link_0": 0.85, "link_9": 0.85}  
        prior_tip_link = "link_0"

    elif mobility_id == "7221":
        sphere_params = {"link_0": 25, "link_1": 32}  
        scale_params = {"link_0": 0.95, "link_1": 0.9}  
    
    elif mobility_id == "11622":
        sphere_params = {"link_0": 20, "link_1": 15}  
        scale_params = {"link_0": 0.88, "link_1": 0.9}     

    elif mobility_id == "48413":
        sphere_params = {"link_0": 15, "link_1": 10}  
        scale_params = {"link_0": 0.88, "link_1": 0.85}     

    elif mobility_id == "7179":
        # 原始的urdf不对，有几个应该是revolute joint(有lower和upper limit)变成了fixed
        sphere_params = {"link_3": 6, "link_5": 15}  
        scale_params = {"link_3": 0.92, "link_5": 0.82}  
        prior_tip_link = "link_3"   

    elif mobility_id == "103013":
        # 原先的init_state.npz不对(通用毛病，joint0和1反了，由于是primistic joint，所以angle要除以scaling factor)
        sphere_params = {"link_0": 8, "link_1": 4}  
        scale_params = {"link_0": 0.85, "link_1": 0.88}

    elif mobility_id == "101380":
        sphere_params = {"link_0": 6, "link_1": 10}  
        scale_params = {"link_0": 0.9, "link_1": 0.95}

    elif mobility_id == "7167":
        sphere_params = {"link_0": 6, "link_3": 4}  
        scale_params = {"link_0": 0.92, "link_3": 0.9}
        prior_tip_link = "link_0" 

    elif mobility_id == "45850":
        sphere_params = {"link_0": 4, "link_1": 2}  
        scale_params = {"link_0": 0.95, "link_1": 0.95}
    # **
    elif mobility_id == "103634":
        sphere_params = {"link_0": 2, "link_3": 2}  
        scale_params = {"link_0": 0.9, "link_3": 0.9}
        prior_tip_link = "link_0" 

    elif mobility_id == "7290":
        sphere_params = {"link_0": 10, "link_1": 15}  
        scale_params = {"link_0": 0.9, "link_1": 0.85}
    # **
    elif mobility_id == "12531":
        sphere_params = {"link_0": 4, "link_1": 4}  
        scale_params = {"link_0": 0.9, "link_1": 0.95}
    # **
    elif mobility_id == "12605":
        sphere_params = {"link_0": 10, "link_1": 3}  
        scale_params = {"link_0": 0.92, "link_1": 0.95}

    elif mobility_id == "7310":
        sphere_params = {"link_0": 10, "link_1": 35}  
        scale_params = {"link_0": 0.92, "link_1": 0.85}

    elif mobility_id == "45372":
        sphere_params = {"link_0": 10, "link_1": 8}  
        scale_params = {"link_0": 0.95, "link_1": 0.89}

    # urdf里axis不是整数(-0.99999999999)，导致self_collision计算报错
    elif mobility_id == "46408":
        sphere_params = {"link_0": 10, "link_1": 15}  
        scale_params = {"link_0": 0.95, "link_1": 0.92}

    # 原先的init_state.npz不对, joint4和5反了
    elif mobility_id == "12428":
        sphere_params = {"link_4": 4, "link_5": 5}  
        scale_params = {"link_4": 0.9, "link_5": 0.85}
        prior_tip_link = "link_4" 

    elif mobility_id == "12543":
        sphere_params = {"link_0": 1, "link_1": 2}  
        scale_params = {"link_0": 0.9, "link_1": 0.88}

    elif mobility_id == "12553":
        sphere_params = {"link_0": 4, "link_2": 6}  
        scale_params = {"link_0": 0.9, "link_2": 0.88}
        prior_tip_link = "link_0" 

    elif mobility_id == "12559":
        sphere_params = {"link_0": 1, "link_1": 2}  
        scale_params = {"link_0": 0.9, "link_1": [0.84, 0.85, 0.85]}

    # 和其他模型不同，link_0是parent link
    elif mobility_id == "12579":
        sphere_params = {"link_0": 1, "link_1": 1}  
        scale_params = {"link_0": 0.86, "link_1": 0.9}

    elif mobility_id == "12580":
        sphere_params = {"link_0": 10, "link_1": 8}  
        scale_params = {"link_0": 0.8, "link_1": [0.83, 0.86, 0.86]}

    # 和其他模型不同，link_0是parent link
    elif mobility_id == "12583":
        sphere_params = {"link_0": 8, "link_1": 25}  
        scale_params = {"link_0": [0.85, 0.85, 0.845], "link_1": [0.85, 0.85, 0.825]}

    elif mobility_id == "12596":
        sphere_params = {"link_0": 15, "link_1": 2}  
        scale_params = {"link_0": 0.9, "link_1": [0.86, 0.87, 0.87]}

    # 和其他模型不同，link_0是parent link
    elif mobility_id == "12614":
        sphere_params = {"link_0": 1, "link_1": 1}  
        scale_params = {"link_0": 0.87, "link_1": 0.95}

    elif mobility_id == "7263":
        sphere_params = {"link_0": 8, "link_19": 58}  
        scale_params = {"link_0": 0.87, "link_19": 0.86}
        prior_tip_link = "link_0"

    # 和其他模型不同，link_0是parent link
    elif mobility_id == "46107":
        sphere_params = {"link_0": 4, "link_1": 8}  
        scale_params = {"link_0": 0.9, "link_1": 0.9}

    elif mobility_id == "46889":
        sphere_params = {"link_0": 10, "link_1": 12}  
        scale_params = {"link_0": 0.88, "link_1": 0.88}

    elif mobility_id == "48167":
        sphere_params = {"link_0": 1, "link_1": 4}  
        scale_params = {"link_0": 0.88, "link_1": 0.88}

    # 建议去掉
    elif mobility_id == "102171":
        sphere_params = {"link_0": 10, "link_1": 12}  
        scale_params = {"link_0": 0.88, "link_1": 0.88}

    elif mobility_id == "103010":
        sphere_params = {"link_2": 6, "link_3": 6}  
        scale_params = {"link_2": [0.87, 0.8, 0.87], "link_3": [0.82, 0.85, 0.8]}
        prior_tip_link = "link_2"

    elif mobility_id == "46452":
        sphere_params = {"link_0": 2, "link_1": 4, "link_2": 4}  
        scale_params = {"link_0": 0.8, "link_1": 0.88, "link_2": 0.88}
        prior_tip_link = "link_2"  

    elif mobility_id == "46456":
        sphere_params = {"link_0": 6, "link_1": 8, "link_2": 8}  
        scale_params = {"link_0": 0.88, "link_1": 0.88, "link_2": [0.85, 0.92, 0.85]}
        prior_tip_link = "link_1"

    else:
        print(f"Model_{mobility_id} is not supported")
        sys.exit(0)

    return sphere_params, scale_params, prior_tip_link


def modify_centroid(mobility_id, centroid_matrix, link_name):

    if mobility_id == "103634" and link_name == "link_0":
        centroid_matrix[1][3]+=0.01

    if mobility_id == "12531" and link_name == "link_0":
        centroid_matrix[2][3]+=0.01

    if mobility_id == "12605" and link_name == "link_0":
        centroid_matrix[1][3]-=0.02

    if mobility_id == "12428":
        if link_name == "link_5":
            centroid_matrix[1][3]+=0.03    # back
            centroid_matrix[2][3]+=0.025   # up
        if link_name == "link_4":
            centroid_matrix[1][3]+=0.01    # back

    if mobility_id == "12543" and link_name == "link_0":
        centroid_matrix[0][3]+=0.02        # right

    if mobility_id == "12559" and link_name == "link_0":
        centroid_matrix[2][3]-=0.01        # back
        centroid_matrix[1][3]-=0.01        # down

    if mobility_id == "12579" and link_name == "link_1":
        centroid_matrix[2][3]+=0.01        # front

    if mobility_id == "12583" and link_name == "link_1":
        centroid_matrix[2][3]+=0.025         

    if mobility_id == "12596" and link_name == "link_0":
        centroid_matrix[1][3]-=0.035  
        centroid_matrix[2][3]+=0.035       

    if mobility_id == "7263" and link_name == "link_0":
        centroid_matrix[2][3]-=0.01   

    if mobility_id == "46107" and link_name == "link_1":
        centroid_matrix[2][3]-=0.012    

    if mobility_id == "48167" and link_name == "link_0":
        centroid_matrix[2][3]-=0.01   

    if mobility_id == "102171":
        if link_name == "link_1":
            centroid_matrix[1][3]+=0.02   
        if link_name == "link_0":
            centroid_matrix[1][3]+=0.03
            centroid_matrix[2][3]-=0.03

    if mobility_id == "103010":
        if link_name == "link_3":
            centroid_matrix[2][3]+=0.03  
        if link_name == "link_2":
            centroid_matrix[1][3]+=0.025  

    if mobility_id == "46452" and link_name == "link_0":
        centroid_matrix[2][3]-=0.02   

    if mobility_id == "46456" and link_name == "link_1":
        centroid_matrix[2][3]-=0.01   