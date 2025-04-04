# Frontend program predicting stereotactic targets from inputed anatomical fiducials for brain imaging applications

*This application is currently in development*

**To run the development environment:**

1. Git clone the stareotactic_target_pred repository `https://github.com/mackenziesnyder/stereotactic-target-pred.git`
2. Set up python backend via `poetry shell`
3. Install the required libraries via `poetry install`
4. Run the backend with `python3 stereotactic_target_pred/backend/app.py`
5. Set up the frontend with `cd stereotactic_target_pred/frontend`
6. Use the latest version of node with `nvm use --lts`
7. Install the required libraries via `npm install`
8. Run the frontend with `npm run dev`

**To run in build environment locally**
1. to build the frontend run `npm run build`
3. `poetry run gunicorn -w 4 -b 127.0.0.1:5001 app:app`
