# Visual Dialogue question asking offline RL environment

To serve the Visual Dialogue question asking environment used by the paper ["Offline RL for Natural Language Generation with Implicit Language Q Learning"](https://sea-snell.github.io/ILQL_site/), follow the steps below:

### Setup
1. `git clone https://github.com/Sea-Snell/visdial-rl.git`
2. `cd visdial-rl`
3. install conda
4. conda create --name my_visdial_env python=3.6.12
5. conda activate my_visdial_env
6. conda install pytorch=0.4.1 -c pytorch
7. `pip install -r requirements.txt`
8. `sudo apt-get update`
9. `sudo apt-get install redis`
10. `redis-server --daemonize yes`
11. Download the zip files from the Google drive folder [here](https://drive.google.com/drive/folders/1TAgja4bF5PyAV6gA5UzEAld2Vyk_qb15?usp=sharing). Place the downloaded and unzipped files, "data" and "checkpoints", at the root of the repo.

### Serve
``` shell
python serve_model.py -useGPU -startFrom checkpoints/abot_sl_ep60.vd -qstartFrom checkpoints/qbot_sl_ep60.vd
```

Optionally remove the `-useGPU` flag if you don't want to serve the models on a GPU.

That's it! The Visual Dialogue environment should be accessible on port 5000 on localhost.
