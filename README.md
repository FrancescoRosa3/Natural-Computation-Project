# Natural-Computation-Project

This project has been realized by:
- Demetrio Trimarco, 0622701179, d.trimarco1@studenti.unisa.it
- Francesco Rosa, 0622701095, f.rosa5@studenti.unisa.it
- Giovanni Mignone, 0622701104, g.mignone2@studenti.unisa.it

## How to run the custom_controller_overtake

Our controller has been optimized in **race** mode.
In this mode the controller need the track-info file.
Let's generate the track-info file.

```bash
wtorcs.exe -t 2000000000 -nofuel
```
Configure the track on which you want to run the contreller, without opponents.
Open another terminal and run:
```bash
python custom_controller_overtake -s 0 -t track_name -p port
```
The value of **port** must be the same as the scr_server plus 1.
For example, if you configure for scr_server_0, you have insert **-p 1**
The value of **track_name** is the name of the track configured. This is used as name of the ***.trackinfo** file.

After the ***.trackinfo** file has been generated, configure the race with the opponents and run:
```bash
python custom_controller_overtake -s 2 -t track_name -p port
```
Where track_name is the same name of the ***.trackinfo** file, without the extension.
With the same warings as above.

## How to evaluate the custom_controller_overtake
If you want to evaluate the custom_controller_overtake with the cost function explaned in the **documentation**, you must run the following commands:

If you do not have the ***.trackinfo** follow the same rules as above.

If you already have the ***.trackinfo**, run
```bash
python evaluate_final_controller.py -s 2 -t track_name -p port
```