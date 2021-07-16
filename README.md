# Natural-Computation-Project

This project has been realized by:
- Demetrio Trimarco, 0622701179, d.trimarco1@studenti.unisa.it
- Francesco Rosa, 0622701095, f.rosa5@studenti.unisa.it
- Giovanni Mignone, 0622701104, g.mignone2@studenti.unisa.it

## Repository content
This repository contains:
### Controller
- **custom_controller_overtake.py**: the controller which implements our overtaking logic.
- **Controller** folder: contains the **optimized parameters** for the final controller.<br><br>
### Others
- **Baseline_snakeoil** folder: contains the snakeoil controller (snakeoil.py) and the baseline controller (client.py).
- **configuration_file** folder: contains the *.xml configuration file used during the training and evaluation of the controller.
- **Parameters_change_condition_files** folder: contains the file used during the training in order to select which parameter must be optimized.
- **Track_info**: contains the ***.trackinfo** files used during the training and the evaluation.
- **custom_controller.py**: the baseline controller with some modification explained in **documentation**.
- **evaluate_custom_controller_overtake_10_runs.py**: file that implements the evaluation process (average of 10 runs) on custom_controller_overtake. 
- **evaluate_custom_controller_overtake_30_runs_std.py**: file that implements the evaluation process (average of 30 runs) on custom_controller_overtake, and compute the standard deviation. 
- **evaluate_custom_controller.py**: file that implements the evaluation process (average of 10 runs) on custom_controller.
- **evaluate_final_controller.py**: file that implements the evaluation of the final controller.
- **PSO&#46;py**: file used during the optimization with the ***Particle Swarm Optimization Algorithm***.
- **DE&#46;py**: file used during the optimization with the ***Differential Evolutional Algorithm***.

## How to run the custom_controller_overtake

Our controller has been optimized in **race** mode.
In this mode the controller needs the track-info file.
Let's generate the track-info file.

```bash
wtorcs.exe -t 2000000000 -nofuel
```
In order to run the controller you must first make a **warm-up run**, to produce ***.trackinfo** file useful for the actual **race run**.
Configure the track without opponents.
Open another terminal and run:
```bash
python custom_controller_overtake -s 0 -t track_name -p port
```
The value of **port** must be the same as the scr_server plus 1.
For example, if you configure for scr_server_0, you have to insert **-p 1**.<br>
The value of **track_name** is the name of the track configured. This is used as name of the ***.trackinfo** file.<br>
After the ***.trackinfo** file has been generated, configure the race with the opponents and run:
```bash
python custom_controller_overtake -s 2 -t track_name -p port
```
Where **track_name** is the same name of the ***.trackinfo** file, without the extension.
With the same warnings as above.

## How to evaluate the custom_controller_overtake
If you want to evaluate the custom_controller_overtake with the cost function explained in the **documentation**, you must run the following commands:

If you do not have the ***.trackinfo** follow the same rules as above.

If you already have the ***.trackinfo**, run
```bash
python evaluate_final_controller.py -s 2 -t track_name -p port
```