import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from obspy.geodetics import gps2dist_azimuth
import math
from collections import defaultdict
import pandas as pd
import copy
import warnings

# filter warnig to be error
warnings.filterwarnings('ignore')
"""
Code for calculate incidence angle using Shooting methode
"""

# build layer function
def build_model(n_top, n_bottom, velocity):
    max_depth = -30000 # set max depth (depth marked with negative sign)
    model = list()
    lay = 0
    for top , bottom in zip (n_top, n_bottom):
        thick = bottom - top
        model.append([top, thick, velocity[lay]])
        lay+=1
        if  bottom < max_depth:
            break
    # rect = [top, thickness, velocity]
    return model

# initiate model by station elevation and the hypocenter postiiton
def upward_model (hypo_depth, sta_elev, model_raw):
    # correct upper model boundary and last layer thickness
    sta_pos = -1
    hypo_pos = -1
    for layer in model_raw:
        if layer[0] >= max(sta_elev, hypo_depth):
            sta_pos+=1
            hypo_pos+=1
        elif layer[0] >= hypo_depth:
            hypo_pos+=1
        else:
            pass
    modified_model = model_raw[sta_pos:hypo_pos+1]
    modified_model[0][0] = sta_elev
    if len(modified_model) > 1:
        modified_model[0][1] = float(modified_model[1][0]) - sta_elev
        modified_model[-1][1] = hypo_depth - modified_model[-1][0]
    else:
        modified_model[0][1] =  hypo_depth - sta_elev
    upward_model = modified_model
    return upward_model
    
def downward_model(hypo_depth, sta_elev, model_raw):
    hypo_pos = -1
    for layer in model_raw:
        if layer[0] >= hypo_depth:
            hypo_pos+=1
    modified_model = model_raw[hypo_pos:]
    modified_model[0][0] = hypo_depth
    if len(modified_model) > 1:
        modified_model[0][1] = float(modified_model[1][0]) - hypo_depth
    downward_model = modified_model
    return downward_model
    
# calculate the shooting method (snell's law)
def up_refract (epi_dist, model_raw, angle):
    # start calculate ray path
    # angle must be a list
    degrees = angle
    holder = defaultdict(dict)
    last_take_off = str
    for degree in degrees:
        holder[f"take_off_{degree}"]= {'emit_degree':[], 'distance':[], 'tt':[]}
        start_dist = 0
        degree_emit = degree
        for i,j in zip(range(1, len(model_raw)+1), range(2, len(model_raw)+2)):
            i=-1*i; j= -1*j           
            dist = np.tan(degree_emit*math.pi/180)*abs(model_raw[i][1])  # abs since the thickness is in negative
            tt = abs(model_raw[i][1])/(np.cos(degree_emit*math.pi/180)*(model_raw[i][-1]*1000)) 
            start_dist += dist
            if start_dist > epi_dist:
                break
            holder[f"take_off_{degree}"]['emit_degree'].append(degree_emit)
            holder[f"take_off_{degree}"]['distance'].append(start_dist)
            holder[f"take_off_{degree}"]['tt'].append(tt)
            try:
                degree_emit = 180*(np.arcsin(np.sin(degree_emit*math.pi/180)*model_raw[j][-1]/model_raw[i][-1]))/math.pi 
            except Exception as e:
                break
        if start_dist > epi_dist:
            break
        last_take_off = degree
    return dict(holder), last_take_off
      
# down refracted 
def down_refract(epi_dist,  model_up, model_down):
    half_dist = epi_dist/2
    # find all the critical angle in the model
    c_angle = list()
    if len(model_down) > 1:
        for i in range(0, len(model_down)):
            j = i+1
            try:
                c_a = 180*(np.arcsin(model_down[i][-1]/model_down[j][-1]))/math.pi
                c_angle.append(c_a)
            except Exception as e:
                pass
    
    # find the first take off angle for every critical angle
    holder_c = defaultdict(list)
    for i in range(0, len(c_angle)):
        if i < 1:
            holder_c[i].append(c_angle[i])
        if i > 0:
            deg_em = c_angle[i]
            v = i + 1
            for k, j in zip (range(-v,0), range(-v+1,0)) :
                k *=-1; j*=-1
                deg_em =  180*(np.arcsin(np.sin(deg_em*math.pi/180)*model_down[j-1][-1]/model_down[k-1][-1]))/math.pi
                holder_c[i].append(deg_em)
                holder_c[i].sort()
            holder_c[i].append(c_angle[i])
    degrees = [holder_c[i][0] for i in list(holder_c.keys())]
    degrees.sort()
    #degrees = np.linspace(28, c_angle, 4)
    #spesi = [41.63971879245696, 49.19710378983109]
    holder_up = defaultdict(dict)
    holder_down = defaultdict(dict)
    for degree in degrees:
        start_dist = 0
        holder_down[f"take_off_{degree}"]= {'emit_degree':[], 'distance':[], 'tt':[]}
        degree_emit = degree
        for i,j in zip(range(0, len(model_down)), range(1, len(model_down)+1)):
            dist = np.tan(degree_emit*math.pi/180)*abs(model_down[i][1])  # abs since the thickness is in negative
            tt = abs(model_down[i][1])/(np.cos(degree_emit*math.pi/180)*(model_down[i][-1]*1000)) 
            start_dist += dist
            if start_dist > half_dist:
                break
            holder_down[f"take_off_{degree}"]['emit_degree'].append(degree_emit)
            holder_down[f"take_off_{degree}"]['distance'].append(start_dist)
            holder_down[f"take_off_{degree}"]['tt'].append(tt)
            
            # cek emit degree ray
            emit_deg = np.sin(degree_emit*math.pi/180)*model_down[j][-1]/model_down[i][-1]
            if emit_deg < 1:
                degree_emit = 180*(np.arcsin(np.sin(degree_emit*math.pi/180)*model_down[j][-1]/model_down[i][-1]))/math.pi
                continue
            
            elif emit_deg == 1:
                degree_emit = 90
                start_degree = holder_down[f"take_off_{degree}"]['emit_degree'][0]
                angle = []                  #function input must be in a list object
                angle.append(start_degree)
                ray_up, last_take_off = up_refract(epi_dist, model_up, angle)
                holder_up.update(ray_up)
                try:
                    dist_up = ray_up[f'take_off_{start_degree}']['distance'][-1]
                    dist_critical = (2*half_dist) - (2*start_dist) - dist_up   # total flat line length
                except Exception as e:
                    continue
                if dist_critical < 0:
                    continue
                tt_critical = (dist_critical / (model_down[j][-1]*1000))
                holder_down[f"take_off_{degree}"]['emit_degree'].append(degree_emit)
                holder_down[f"take_off_{degree}"]['distance'].append(dist_critical + start_dist)
                holder_down[f"take_off_{degree}"]['tt'].append(tt_critical)
                break
                
            else:
                break
    return holder_up, holder_down


def calculate_inc_angle(hypo, sta, model, plot_figure = True):
    [hypo_lat,hypo_lon, depth] = hypo
    [sta_lat, sta_lon, elev] = sta
    [top, bottom, velocity] = model
    epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)
    
    # build raw model
    model_raw = build_model(top, bottom, velocity)
    
    # deep copy the model (original copy is  needed for plotting)
    model_copied = copy.deepcopy(model_raw)
    model_copied_2 = copy.deepcopy(model_raw)
    up_model = upward_model (depth, elev, model_copied)
    down_model = downward_model(depth, elev, model_copied_2)
    
    # build upward and downward model
    angle_resolution = np.linspace(0, 90, 1000)
    up_ref, last_take_off = up_refract(epicentral_distance, up_model, angle_resolution)
    up_ray, down_ray = down_refract(epicentral_distance, up_model, down_model)
    
    # calculate the result
    last_ray = up_ref[f"take_off_{last_take_off}"]
    take_off_upward_refract = 180 - last_ray['emit_degree'][0]
    upward_refract_tt = np.sum(last_ray['tt'])
    reach_distance = last_ray['distance'][-1] # -1 index since distance is cumulative value
    upward_incidence_angle = last_ray['emit_degree'][-1]

    # calculate total travel time of downward travel time
    checker_take_off = list()
    for key in down_ray.keys():
        try:
            if down_ray[key]['emit_degree'][-1] == 90:
                first_emit_degree = down_ray[key]['emit_degree'][0]
                checker_take_off.append(first_emit_degree)
        except Exception as e:
            pass
            
    refracted_ray = defaultdict(dict) # list of downard refracted ray (take_off_angle, total_tt, incidence_angle)
    if len(checker_take_off):
        for take_off in checker_take_off:
            if len(up_ray[f"take_off_{take_off}"]['distance']) < len(last_ray['distance']):  # only take the reach signal
                continue
            refracted_ray[f'{take_off}'] = {'total_tt':[], 'incidence_angle': []}
            tt_downward = np.sum(down_ray[f"take_off_{take_off}"]['tt'])
            tt_upward_sec_half = np.sum(down_ray[f"take_off_{take_off}"]['tt'][:-1])
            tt_refract_upward = np.sum(up_ray[f"take_off_{take_off}"]['tt'])
            total_tt = tt_downward + tt_upward_sec_half + tt_refract_upward
            incidence_angle = up_ray[f"take_off_{take_off}"]['emit_degree'][-1]
            refracted_ray[f'{take_off}']['total_tt'].append(total_tt)
            refracted_ray[f'{take_off}']['incidence_angle'].append(incidence_angle)
            
    if len(refracted_ray): 
        fastest_ray = 9999
        fastest_key = str
        for key in refracted_ray.keys():
            if refracted_ray[key]['total_tt'][-1] < fastest_ray:     # find the fastest ray from downward refraction
                fastest_ray = refracted_ray[key]['total_tt'][-1]
                fastest_key = key

        if refracted_ray[fastest_key]['total_tt'][-1] > upward_refract_tt:
            takeoff = take_off_upward_refract
            total_travel = upward_refract_tt
            incidence_angle = upward_incidence_angle

        else:
            takeoff = fastest_key
            total_travel = refracted_ray[fastest_key]['total_tt'][-1]
            incidence_angle = refracted_ray[fastest_key]['incidence_angle'][-1]
    else:
        takeoff = take_off_upward_refract
        total_travel = upward_refract_tt
        incidence_angle = upward_incidence_angle
    
    
    if plot_figure:
        # for plotting
        fig, axs = plt.subplots()
        # making colormaps
        cmap = cm.Oranges
        norm = mcolors.Normalize(vmin=min(velocity), vmax=max(velocity))
        max_depth = 0
        max_width = epicentral_distance + 2000
        for layer in model_1:
            color = cmap(norm(layer[-1]))
            rect = patches.Rectangle((-1000, layer[0]), max_width, layer[1], linewidth=1, edgecolor= 'black', facecolor = color)
            axs.add_patch(rect)
            max_depth +=layer[1]
        
        # for plotting only the last ray reach the station
        layer_cek = 0
        x1 = 0 
        for layer in reversed(up_model):
            if layer_cek == 0:
                y1 = layer[0] + layer[1]
            x2 = last_ray['distance'][layer_cek]
            y2 = layer[0]
            axs.plot([x1,x2], [y1,y2], color = 'k')
            x1=x2
            y1=y2
            layer_cek+=1


        # for plotting only rays that reach the station
        if len(refracted_ray):
            for take_2 in refracted_ray.keys():
                layer_cek = 0
                try:
                    x1 =  0
                    for layer in down_model:
                        if layer_cek == 0:
                            y1 = layer[0]
                        x2 = down_ray[take_2]['distance'][layer_cek]
                        
                        # plot second half
                        if down_ray[take_2]['emit_degree'][layer_cek] == 90:
                            y2 = layer[0]
                            axs.plot([x1,x2], [y1,y2], color = 'b')
                            
                            # plot the upward of the second half 
                            up_check = layer_cek
                            d1 = x2
                            r1 = y2
                            for layer in reversed(down_model[:layer_cek]):
                                r2 = layer[0]
                                d2 =  d1 + down_ray[take_2]['distance'][up_check - 1] - down_ray[take_2]['distance'][up_check - 2]
                                if up_check == 1:
                                    d2 =  d1 + down_ray[take_2]['distance'][up_check - 1]
                                axs.plot([d1,d2], [r1,r2], color = 'b')
                                d1 = d2
                                r1 = r2
                                up_check -= 1

                            # plotting up refract
                            model_up_cek = 0
                            q1 = d2
                            for layer in reversed(up_model):
                                if model_up_cek == 0:
                                    t1 = layer[0] + layer[1]
                                q2 = q1 + up_ray[take_2]['distance'][model_up_cek] - up_ray[take_2]['distance'][model_up_cek - 1]
                                if model_up_cek == 0:
                                    q2 = q1 + up_ray[take_2]['distance'][model_up_cek]
                                t2 = layer[0] 
                                axs.plot([q1,q2], [t1,t2], color = 'b')
                                q1=q2
                                t1=t2
                                model_up_cek+=1
                            break
                        y2 = layer [0] + layer [1]
                        axs.plot([x1,x2], [y1,y2], color = 'b')
                        x1 = x2
                        y1 = y2
                        layer_cek +=1
                except Exception as e:
                    #print(e)
                    pass
        #report
        print("Upward refracted solutions:\n",
                "Take off angle :",take_off_upward_refract,'\n',
                "Incidence angle:", upward_incidence_angle,'\n',
                "Travel time:", upward_refract_tt, '\n',
                "Reached distance:", reach_distance, '\n'
                )                
        
        if len(refracted_ray): 
            print("\nTakeoff, incidence angle, and total travel time from the refracted downward ray that reach the station distance:")
            fastest_ray = 9999
            fastest_key = str
            for key in refracted_ray.keys():
                print('\t', key, refracted_ray[key])
                if refracted_ray[key]['total_tt'][-1] < fastest_ray:     # find the fastest ray from downward refraction
                    fastest_ray = refracted_ray[key]['total_tt'][-1]
                    fastest_key = key
            
            print("\nFastest downward refracted ray:",'\n',
                    f"Take off angle : {fastest_key} \n",
                    f"Total travel time: {refracted_ray[fastest_key]['total_tt'][-1]}\n",
                    f"Incidende angle: {refracted_ray[fastest_key]['incidence_angle'][-1]}")

            print("\nFinal shooting snell solutions:")
            if refracted_ray[fastest_key]['total_tt'][-1] > upward_refract_tt:
                print(
                    f"Take off angle : {take_off_upward_refract} \n",
                    f"Total travel time: {upward_refract_tt}\n",
                    f"Incidende angle: {upward_incidence_angle}")
            else:
                print(
                    f"Take off angle : {fastest_key} \n",
                    f"Total travel time: {refracted_ray[fastest_key]['total_tt'][-1]}\n",
                    f"Incidende angle: {refracted_ray[fastest_key]['incidence_angle'][-1]}")

        else:
            print("\nNo downward refracted rays reach the station are available....")
        max_width = epicentral_distance + 2000
        #plot the source and the station
        axs.plot(epicentral_distance, elev, marker = 'v', color = 'black', markersize = 15)
        axs.text(epicentral_distance , elev + 200 , station_name, horizontalalignment='right', verticalalignment='center')
        axs.plot(0, depth, marker = '*', color = 'red', markersize = 12)
        axs.set_xlim(-2000,max_width)
        axs.set_ylim(-3000, 3000)
        axs.set_ylabel('Depth')
        axs.set_xlabel('Distance')
        axs.set_title('Shooting Snell Method')
        plt.show()
        
    return takeoff, total_travel, incidence_angle

# End of functions 
#============================================================================================
if __name__ == "__main__" :
    
    # Parameters
    # model parameters
    n_top =    [3000, 1900, 590, -220, -2500, -7000, -9000, -15000, -33000]
    n_bottom = [1900, 590, -220, -2500, -7000, -9000, -15000, -33000, -99999]
    velocity = [2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00]
    model_raw = [n_top, n_bottom, velocity]
    
    # hypo and station
    hypo_3 = [-4.2043831, 103.4367594, 805.34]  # depth is in negative (downward direction)
    hypo_2 = [-4.2148923, 103.4049723, -757.7]
    hypo_1 = [-4.2158183, 103.3982992, -187.65]
    hypo =  [-4.2049606, 103.4356201, 633.69]
    hypo_4 = [-4.2040105, 103.4403158, 1162.44]
    hypo_test = [-4.2181541, 103.3998372, -193.84]
    [hypo_lat,hypo_lon, depth] = hypo_test

    rd2 =[-4.210773, 103.401842, 1753]
    rd15 = [-4.243717, 103.369522, 2610]
    rd12 = [-4.236119, 103.366364, 2467]
    rd6 = [-4.212581, 103.379951, 1943]
    rd5 = [-4.20397, 103.378906, 2013]
    rd10 = [-4.231261, 103.359078, 2412]
    rd1 = [, -4.192293, 103.408419, 1504]
    rd4 = [ -4.210271, 103.417561, 1719]
    rd3 = [-4.20593, 103.390728, 1899]
    [station_name, sta_lat, sta_lon, elev] = rd3
    
    
    ray_inc = calculate_inc_angle(hypo_test, rd10, model)