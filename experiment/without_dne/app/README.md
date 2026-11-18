# Experiment Runner

This folder contains only the files necessary for **running experiments**, not for analyzing data.

## Structure

```
experiment_runner/
├── __init__.py              # Flask app initialization
├── experiment.py            # Main experiment routes
├── consent.py               # Consent form
├── alert.py                 # Alert/instructions
├── complete.py              # Completion page
├── error.py                 # Error handling
├── io.py                    # Data I/O utilities
├── utils.py                 # Utility functions
├── app.ini                  # Configuration
├── templates/               # HTML templates
├── static/                  # Frontend assets
│   ├── js/                  # JavaScript files
│   ├── lib/                  # Libraries (jspsych, jquery)
│   ├── css/                  # Stylesheets
│   └── img/                  # Images
└── os_model/                # Machine learning models
    ├── data_loader.py       # Data loading utilities
    ├── visualizer.py        # Visualization (minimal)
    ├── exp/                 # Experiment framework
    │   ├── Oneshot.py
    │   ├── Oneshot_sangwan.py
    │   ├── Oneshot_Naive.py
    │   └── Oneshot_Random.py
    └── module_RNN/          # Neural network models
        ├── dnns.py          # LSTM_Attn2 model
        ├── sublayers.py     # Model components
        ├── transformer.py   # Transformer architecture
        └── dne_driving_model.py  # Training loop
```

## What's Included

✅ All Flask application files  
✅ All HTML templates  
✅ All JavaScript files and libraries  
✅ All CSS and image files  
✅ Core ML model files needed for real-time training  
✅ Experiment framework files  

## What's Excluded

❌ Analysis scripts (`*_profiling*.py`, `*_diagnosis.py`)  
❌ Analysis results folders (`model_driving/`, `model_profiling/`)  
❌ Subject analysis scripts (`subject_driving/`, `subject_profiling/`)  
❌ Visualization outputs (`fig_attn/`)  
❌ Test files and shell scripts  

## Running the Experiment

1. Install dependencies (Flask, PyTorch, NumPy, etc.)
2. Configure `app.ini` with your settings
3. Run: `python -m flask run` (or use your WSGI server)
4. Access via: `http://localhost:5000/?workerId=TEST123`

## Data Directories

The app will create these directories at runtime (configured in `app.ini`):
- `../data/` - Complete experiment data
- `../metadata/` - Participant metadata
- `../incomplete/` - Incomplete data + model checkpoints
- `../reject/` - Rejected data

