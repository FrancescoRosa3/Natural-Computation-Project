import json
import xml.etree.ElementTree as ET
import argparse

# Load the custom_controller module
import custom_controller_overtake as custom_controller
# define the path were the parameters are defined
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# CONSTANT DEFINITION
NUMBER_SERVERS = 10
BASE_PORT = 3000
PERCENTAGE_OF_VARIATION = 40

# CONSTANT FOR NORMALIZATION
EXPECTED_NUM_LAPS = 2
MAX_SPEED = 300
FORZA_LENGTH = 5784.10
FORZA_WIDTH = 11.0
WHEEL_LENGHT = 4328.54
WHEEL_WIDTH = 14.0
CG_2_LENGHT = 3185.83
CG_2_WIDTH = 15.0
TRACK_LENGTH = {'forza': FORZA_LENGTH, 'wheel-1': WHEEL_LENGHT, 'g-track-2': CG_2_LENGHT}
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
    if conf_split_underscore[-3] == 'no':
        global adversarial
        adversarial = False
        print("Adversarial {adversarial}")

if __name__ == "__main__":
    ####################### SETUP ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration_file', '-conf', help="name of the configuration file to use, without extension or port number", type= str,
                    default= "quickrace_forza_no_adv")
    parser.add_argument('--controller_params', '-ctrlpar', help="initial controller parameters", type= str,
                    default= "Baseline_snakeoil\default_parameters")
   
                    
    args = parser.parse_args()

    # load default parameters
    args.controller_params = '\\' + args.controller_params
    pfile= open(dir_path + args.controller_params,'r') 
    parameters = json.load(pfile)
    
    # load the change condition file
    get_configuration(args.configuration_file)    

    track_names = take_track_names(args)

    fitness_dict_component = {}
    fitnesses_dict = {}
    for i in range(0, 10):
        for track in track_names:
            try:
                #print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
                controller = custom_controller.CustomController(port=3010,
                                                                parameters=parameters, 
                                                                parameters_from_file=False,
                                                                stage=2,
                                                                track=track)

                history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks, race_failed = controller.run_controller(plot_history = True)

                normalized_ticks = ticks/controller.C.maxSteps

                # compute the number of laps
                num_laps = len(history_lap_time)

                # the car has completed at least the first lap
                if num_laps > 0:
                    # compute the average speed
                    avg_speed = 0
                    for history_key in history_speed.keys():
                        for value in history_speed[history_key]:
                            avg_speed += value
                    avg_speed /= ticks
                    #print(f"Num Laps {num_laps} - Average Speed {avg_speed} - Num ticks {ticks}")
                    
                    normalized_avg_speed = avg_speed/MAX_SPEED

                    distance_raced = history_distance_raced[history_key][-1]
                    normalized_distance_raced = distance_raced/(TRACK_LENGTH[track]*EXPECTED_NUM_LAPS)

                    # take the damage
                    damage = history_damage[history_key][-1]
                    normalized_damage = damage/UPPER_BOUND_DAMAGE if not adversarial else damage/UPPER_BOUND_DAMAGE_WITH_ADV

                    # take the car position at the end of the race
                    car_position = history_car_pos[history_key][-1]
                    car_position -= 1
                    norm_car_position = car_position/OPPONENTS_NUMBER

                    # compute the average from the center line
                    """
                    average_track_pos = 0
                    steps = 0
                    for key in history_track_pos.keys():
                        for value in history_track_pos[key]:
                            steps += 1
                            if abs(value) > 1:
                                average_track_pos += (abs(value) - 1)
                    average_track_pos /= steps
                    """

                    # compute out of track ticks and normilize it with respect to the total amount of ticks
                    ticks_out_of_track = 0
                    for key in history_track_pos.keys():
                        for value in history_track_pos[key]:
                            if abs(value) > 1:
                                ticks_out_of_track += 1
                    norm_out_of_track_ticks = ticks_out_of_track/MAX_OUT_OF_TRACK_TICKS                    
                    
                    # compute the fitness for the current track
                    speed_comp_multiplier = 2
                    car_pos_multiplier = 2
                    fitness = (-normalized_avg_speed * speed_comp_multiplier) -normalized_distance_raced +normalized_damage +norm_out_of_track_ticks +normalized_ticks + (norm_car_position * car_pos_multiplier)
                    # store the fitness for the current track
                    fitness_dict_component[track] = f"Fitness {fitness:.4f}\nCar position {norm_car_position:.4f}\nNorm AVG SPEED {-normalized_avg_speed:.4f}\nNorm Distance Raced {-normalized_distance_raced:.4f}\nNorm Damage {normalized_damage:.4f}\nnorm out_of_track_ticks {norm_out_of_track_ticks:.4f}\nnormalized ticks {normalized_ticks:.4f}\nSim seconds {ticks/50}"
                    
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

        print(f"Run: {i}\nTotal fitness {total_fitness}\nFitness components {fitness_dict_component}")
