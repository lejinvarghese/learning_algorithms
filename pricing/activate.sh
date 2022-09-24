conda activate rl-pricing
conda install tensorflow-gpu==2.2.2 cudatoolkit=10.1
conda install --file requirements.txt
pip install torch jupyterlab ray[default] torch qbstyles dm_tree torchaudio gym pandas scikit-image tabulate seaborn bokeh

python -m ipykernel install --name "rl-pricing" --user
jupyter lab --notebook-dir=. --port 8080 --ip 0.0.0.0 --ContentsManager.allow_hidden=True
