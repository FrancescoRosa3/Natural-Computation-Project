import json
import xml.etree.ElementTree as ET
import argparse
import numpy as np

# Load the custom_controller module
#import custom_controller_overtake as custom_controller
import custom_controller
# define the path were the parameters are defined
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


# CONSTANT FOR NORMALIZATION
EXPECTED_NUM_LAPS = 2
MAX_SPEED = 300
UPPER_BOUND_DAMAGE = 1500
UPPER_BOUND_DAMAGE_WITH_ADV = 7000
MAX_OUT_OF_TRACK_TICKS = 1000       # corresponds to 20 sec
OPPONENTS_NUMBER = 8

SAVE_CHECKPOINT = True

# list of track names
track_names = []

# boolean for adversarial
adversarial = True

def take_track_names(args):
    track_names = []

    path_name = dir_path + "/configuration_file/" + args.configuration_file + f"_{1}.xml"
    
    configuration_file = ET.parse(path_name)
    root = configuration_file.getroot()

    for child in root:
        if child.attrib['name'] == 'Tracks':
            for section in child.findall('section'):
                for sub_section in section.findall('attstr'):
                    if sub_section.attrib['name'] == 'name':
                        track_names.append(sub_section.attrib['val'])
    return track_names 

def get_configuration(path):
    conf_split_underscore = path.split("_") 
    if conf_split_underscore[-2] == 'no':
        global adversarial
        adversarial = False
        print(f"Adversarial {adversarial}")

if __name__ == "__main__":
    ####################### SETUP ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration_file', '-conf', help="name of the configuration file to use, without extension or port number", type= str,
                    default= "quickrace_forza_no_adv")
    parser.add_argument('--controller_params', '-ctrlpar', help="initial controller parameters", type= str,
                    default= "Baseline_snakeoil\default_parameters")
    parser.add_argument('--port', '-p', help="server port 1-10", type= int,
                    default= 1)
                    
    args = parser.parse_args()

    # load default parameters
    args.controller_params = '\\' + args.controller_params
    pfile= open(dir_path + args.controller_params,'r') 
    parameters = json.load(pfile)
    
    # load the change condition file
    get_configuration(args.configuration_file)    

    track_names = take_track_names(args)

    average_fitness_dict = {}
    for track in track_names:
        if not adversarial:
            average_fitness_dict[track] = {
                                                "fitness": 0.0, "norm_avg_speed":0.0, "norm_out_of_track_ticks": 0.0,
                                                "norm_max_speed": 0.0#, "norm_min_speed": norm_min_speed
                                            }
        else:
            average_fitness_dict[track] = {
                                                "fitness": 0.0, "norm_final_car_position": 0.0,
                                                "norm_best_car_position": 0.0,
                                                "norm_out_of_track_ticks": 0.0, "normalized_damage": 0.0
                                            }



    fitness_dict_component = {}
    fitnesses_dict = {}
    for i in range(0, 10):
        for track in track_names:
            try:
                #print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
                controller = custom_controller.CustomController(port=3000+args.port,
                                                                parameters=parameters, 
                                                                parameters_from_file=False,
                                                                stage=2,
                                                                track=track)

                history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks, race_failed = controller.run_controller()
                while race_failed == True:
                    print("Server crashed, restarting agent...")
                    history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks, race_failed = controller.run_controller()
                        
                normalized_ticks = ticks/controller.C.maxSteps

                # compute the number of laps
                num_laps = len(history_lap_time)

                # the car has completed at least the first lap
                if num_laps > 0:
                    # compute the average speed
                    avg_speed = 0
                    max_speed = 0
                    min_speed = MAX_SPEED
                    for history_key in history_speed.keys():
                        for value in history_speed[history_key]:
                            max_speed = value if value > max_speed else max_speed
                            min_speed = value if value < min_speed else min_speed
                            avg_speed += value
                    avg_speed /= ticks
                    norm_max_speed = max_speed/MAX_SPEED
                    norm_min_speed = min_speed/MAX_SPEED
                    #print(f"Num Laps {num_laps} - Average Speed {avg_speed} - Num ticks {ticks}")
                    
                    normalized_avg_speed = avg_speed/MAX_SPEED
                
                    # take the damage
                    damage = history_damage[history_key][-1]
                    normalized_damage = damage/UPPER_BOUND_DAMAGE if not adversarial else damage/UPPER_BOUND_DAMAGE_WITH_ADV

                    # take the car position at the end of the race
                    final_car_position = history_car_pos[num_laps][-1]
                    final_car_position -= 1
                    norm_final_car_position = final_car_position/OPPONENTS_NUMBER

                    # take the best position during the race
                    best_car_position = 9
                    for lap in range(1, num_laps+1):
                        best_car_position = np.min(history_car_pos[lap]) if np.min(history_car_pos[lap]) < best_car_position else best_car_position
                    best_car_position -= 1
                    norm_best_car_position = best_car_position/OPPONENTS_NUMBER
                    # compute out of track ticks and normilize it with respect to the total amount of ticks
                    ticks_out_of_track = 0
                    for key in history_track_pos.keys():
                        for value in history_track_pos[key]:
                            if abs(value) > 1:
                                ticks_out_of_track += 1
                    norm_out_of_track_ticks = ticks_out_of_track/MAX_OUT_OF_TRACK_TICKS                       
                    
                                            # compute the fitness for the current track and store the fitness for the current track
                    if not adversarial:
                        fitness = - normalized_avg_speed - norm_max_speed + norm_out_of_track_ticks # - norm_min_speed 
                        fitness_dict_component[track] = {
                                                            "fitness": fitness, "norm_avg_speed":-normalized_avg_speed, "norm_out_of_track_ticks": norm_out_of_track_ticks,
                                                            "norm_max_speed": norm_max_speed#, "norm_min_speed": norm_min_speed
                                                        }
                    else:
                        car_pos_multiplier = 2
                        fitness = (norm_final_car_position * car_pos_multiplier) + norm_best_car_position + norm_out_of_track_ticks  + normalized_damage 
                        fitness_dict_component[track] = {
                                                            "fitness": fitness, "norm_final_car_position": norm_final_car_position,
                                                            "norm_best_car_position": norm_best_car_position,
                                                            "norm_out_of_track_ticks": norm_out_of_track_ticks, "normalized_damage": normalized_damage
                                                        }
                else:
                    #print(f"THE AGENTS COULDN'T COMPLETE THE FIRST LAP")
                    fitness = 10  
                #return fitness
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                #print(message)
                fitness = 20

            fitnesses_dict[track] = fitness
            

        total_fitness = 0
        num_track = 0
        for fitness_on_track in fitnesses_dict.keys():
            total_fitness += fitnesses_dict[fitness_on_track]
            num_track += 1
        total_fitness /= num_track

        for track in fitness_dict_component.keys():
            for term in fitness_dict_component[track].keys():
                average_fitness_dict[track][term] += fitness_dict_component[track][term]

        print(f"Run: {i}\nTotal fitness {total_fitness}\nFitness components {fitness_dict_component}")


    for track in average_fitness_dict.keys():
        for term in average_fitness_dict[track].keys():
            average_fitness_dict[track][term] /= 10
    
    print(f"Average: {average_fitness_dict}")


