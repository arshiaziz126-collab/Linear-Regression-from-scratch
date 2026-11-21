# Deploying GlycoCast on Vercel

GlycoCast is one Vercel project with two parts:

- **Backend:** `app.py` (Python, FastAPI). It runs the model and draws the charts.
- **Frontend:** `frontend/` (website). FastAPI serves it with `app.frontend("/", directory="frontend")`,
  and Vercel puts these files on its CDN automatically.

Vercel finds the FastAPI `app` in `app.py` and installs `requirements.txt` (FastAPI, NumPy, matplotlib).
No `vercel.json` is needed. The app runs as a serverless function, so it does **not** go to sleep like
Streamlit Community Cloud.

## 1. Run it on your laptop
```bash
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```
Open http://localhost:8000 for the website, and http://localhost:8000/docs to try the API.

To retrain the model, first install the training extras: `python -m pip install -r requirements-train.txt`,
then run `python train.py`.

## 2. Put it on GitHub
Create an empty repository called `glycocast` on github.com (don't add a README). Then, in this folder:
```bash
git init
git add .
git commit -m "GlycoCast: FastAPI backend + website"
git branch -M main
git remote add origin https://github.com/<your-username>/glycocast.git
git push -u origin main
```

## 3. Deploy on Vercel
1. On vercel.com, choose **Add New → Project**.
2. **Import** the `glycocast` repository.
3. Framework Preset should show **FastAPI** (Vercel detects it). Leave Root Directory as `./`.
4. Click **Deploy**. After a minute you get a link like `glycocast.vercel.app`.

Every `git push` redeploys automatically.

## If something goes wrong
- **Build fails:** open the deployment → **Build Logs** and read the last red lines.
- **The page loads but shows "Could not reach the model":** open **Logs** (runtime logs) in the Vercel project.
- **The first request after a while is a bit slower:** that's a cold start (a second or two), and it's normal.
  The app does not sleep or show a "wake up" screen.
