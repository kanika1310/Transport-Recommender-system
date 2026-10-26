# GitHub guide (first-time upload)

## A. One-time setup
1. Create an account at github.com (use your college email if you want student benefits).
2. Install Git: https://git-scm.com/downloads  then check with `git --version`.
3. Tell Git who you are:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

## B. Create the repository on the website
1. Click **+** (top right) -> **New repository**.
2. Name: `transport-recommender-system`. Description: one line about the project.
3. Choose **Public** (examiners need to open it without logging in).
4. Do NOT tick "Add a README" (you already have one). Click **Create repository**.

## C. Push your folder
Open a terminal inside the project folder:
```bash
git init
git add .
git commit -m "Initial commit: project structure, experiment code and README"
git branch -M main
git remote add origin https://github.com/<your-username>/transport-recommender-system.git
git push -u origin main
```
GitHub asks you to sign in. If the password is rejected, create a **Personal Access Token**
(Settings -> Developer settings -> Tokens) and paste it as the password, or use GitHub Desktop.

No terminal? Use **Add file -> Upload files** on the repository page and drag the folder contents in.

## D. After you have real results
```bash
python src/run_experiments.py --data data/trips.csv
git add data/trips.csv results/ docs/
git commit -m "Add real survey data and experiment results"
git push
```
Commit after each real step (data added, results added, paper added) so the history reflects your actual work.

## E. Checklist before you submit
- [ ] Repository is Public and the link opens in a private/incognito window
- [ ] README has the correct clone URL and your names
- [ ] `data/trips.csv` is the real dataset (or explain in README why it is not included)
- [ ] `results/` contains the figures used in the paper
- [ ] The paper PDF is in `docs/`
- [ ] Teammates added: Settings -> Collaborators
- [ ] Paper contains the exact repo URL (search for `<your-username>` and replace)
