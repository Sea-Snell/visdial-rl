# Visual Dialogue question asking offline RL environment

To serve the Visual Dialogue question asking environment used by the paper "Offline RL for Natural Language Generation with Implicit Language Q Learning", follow the steps below:

### Setup
1. install python 3.6.12
2. create a new python environment
3. `pip install -r requirements.txt`
4. `sudo apt-get update`
5. `sudo apt-get install redis`
6. Download the zip files from the Google drive folder [here](https://drive.google.com/drive/folders/1TAgja4bF5PyAV6gA5UzEAld2Vyk_qb15?usp=sharing). Place the downloaded and unzipped files, "data" and "checkpoints", at the root of the repo.

### Serve
``` shell
python serve_model.py -useGPU -startFrom checkpoints/abot_sl_ep60.vd -qstartFrom checkpoints/qbot_sl_ep60.vd
```

That's it! The Visual Dialogue environment should be accessible on port 5000 on localhost.
