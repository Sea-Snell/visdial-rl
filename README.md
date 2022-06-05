# Visual Dialogue question asking offline RL environment

To serve the Visual Dialogue question asking environment used by the paper ["Offline RL for Natural Language Generation with Implicit Language Q Learning"](https://sea-snell.github.io/ILQL_site/), follow the steps below:

### Setup
1. install conda
2. conda create --name my_visdial_env python=3.6.12
3. conda activate my_visdial_env
4. conda install pytorch=0.4.1 -c pytorch
5. `pip install -r requirements.txt`
6. `sudo apt-get update`
7. `sudo apt-get install redis`
8. `redis-server --daemonize yes`
9. Download the zip files from the Google drive folder [here](https://drive.google.com/drive/folders/1TAgja4bF5PyAV6gA5UzEAld2Vyk_qb15?usp=sharing). Place the downloaded and unzipped files, "data" and "checkpoints", at the root of the repo.

### Serve
``` shell
python serve_model.py -useGPU -startFrom checkpoints/abot_sl_ep60.vd -qstartFrom checkpoints/qbot_sl_ep60.vd
```

Optionally remove the `-useGPU` flag if you don't want to serve the models on a GPU.

That's it! The Visual Dialogue environment should be accessible on port 5000 on localhost.
