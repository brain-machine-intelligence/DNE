from itertools import count
from flask import (Blueprint, current_app, redirect, render_template, request, session, url_for, jsonify)
from .io import write_data, write_metadata
from .dne_log import log_event, get_status_store, save_participant_experiment_log
import json
import time, os
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from app.os_model.data_loader import json_to_nps, load_os_data
from app.os_model.module_RNN.dne_driving_model import iterate_epoch
from app.os_model.module_RNN.sublayers import CosineAnnealingWarmupRestarts
from app.os_model.module_RNN.dnns import LSTM_Attn2

## Initialize blueprint.
bp = Blueprint('experiment', __name__)
total_epoch = 20
# Epochs per train_iter request. Lower = faster response, more requests per session (e.g. 10 or 5 for many participants).
EPOCHS_PER_TRAIN_ITER = 50
# GPU lock timeout (seconds). Only one train_iter uses GPU at a time when multiple workers run.
GPU_LOCK_TIMEOUT = 600

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


# Prefer dynamically detected CUDA devices (respects CUDA_VISIBLE_DEVICES)
def _available_devices():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)]
    return []


def _sanitize_device(device_str: str):
    # fallback to CPU if CUDA index is invalid
    if not isinstance(device_str, str):
        return "cpu"
    if device_str.startswith("cuda:"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            idx = int(device_str.split(":", 1)[1])
        except Exception:
            return "cpu"
        if idx < 0 or idx >= torch.cuda.device_count():
            return "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
        return device_str
    if device_str == "cpu":
        return "cpu"
    # unknown string
    return "cpu"


available_gpus = _available_devices()


def _write_experiment_log_start(participant_dir, worker_id, sub_id):
    """_model_0.pt 저장하는 바로 그 폴더(incomplete/subId)에 experiment_log.txt 생성."""
    from datetime import datetime
    if not participant_dir or not worker_id:
        return
    path = os.path.join(participant_dir, "experiment_log.txt")
    lines = [
        "=== DNE experiment log ===",
        "workerId: %s" % (worker_id,),
        "subId: %s" % (sub_id or ""),
        "start_time: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status: started",
        "",
        "--- errors (20|20 alloc 실패 등) ---",
        "(none)",
        "",
    ]
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        if current_app:
            current_app.logger.warning("[DNE] experiment_log write failed path=%r: %s", path, e)


@bp.route('/experiment')
def experiment():

    """Present jsPsych experiment to participant."""

    ## Error-catching: screen for missing session.
    if not 'workerId' in session:

        log_event("experiment_missing_workerId", reason="no workerId in session")

        ## Redirect participant to error (missing workerId).
        return redirect(url_for('error.error', errornum=1000))

    ## Case 1: previously completed experiment.
    elif 'complete' in session:
        
        log_event("experiment_revisit_complete", complete=session.get("complete"))

        ## Redirect participant to complete page.
        return redirect(url_for('complete.complete'))

    ## Case 2: repeat visit.
    elif not session['allow_restart'] and 'experiment' in session:

        ## Update participant metadata.
        session['ERROR'] = "1004: Revisited experiment."
        session['complete'] = 'error'
        write_metadata(session, ['ERROR','complete'], 'a')
        log_event("experiment_revisit_blocked", error=session['ERROR'])

        ## Redirect participant to error (previous participation).
        return redirect(url_for('error.error', errornum=1004))

    ## Case 3: first visit.
    else:
            
        ## Update participant metadata.
        session['experiment'] = True
        
        incomp_dir = session['incomplete'].replace('\\\\','\\')
        participant_dir = os.path.join(incomp_dir, session['subId'])  # _model_0.pt 와 experiment_log.txt 둘 다 이 폴더
        incomp_file = os.path.join(participant_dir, session['subId']+".json");
        model_file = os.path.join(participant_dir, "_model_{}.pt");
        
        try:
            os.makedirs(participant_dir)
            current_app.logger.info("[DNE] participant_dir created: %s", participant_dir)
        except Exception as e:
            current_app.logger.warning("[DNE] makedirs failed: %s", e)

        session['participant_dir'] = participant_dir  # redirect 시 같은 경로로 로그 저장용

        # 실험 시작 로그: _model_0.pt 저장하는 바로 그 폴더에 experiment_log.txt 생성
        _write_experiment_log_start(participant_dir, session['workerId'], session['subId'])
        try:
            log_event("experiment_log_started", workerId=session['workerId'], subId=session['subId'], path=participant_dir)
        except Exception:
            pass

        session['incomp_file'] = incomp_file
        session['model_file'] = model_file
        session['s_count'] = 0
        # Choose a valid device. Avoid "invalid device ordinal" by sampling only from detected GPUs.
        if len(available_gpus) > 0:
            session['gpu_n'] = str(np.random.choice(available_gpus))
        else:
            session['gpu_n'] = "cpu"
        session['gpu_n'] = _sanitize_device(session['gpu_n'])

        print(session['gpu_n'])
        net = LSTM_Attn2(in_channel=5, dropout=0.0, class_num=2, device=session['gpu_n']);
        lr = 0.01
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                                     cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                                     warmup_steps=150, gamma=0.5)
        net.save_model(fname=session['model_file'].format(session['s_count']), 
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler)
        
        write_metadata(session, ['experiment'], 'a')
        del net
        del optimizer
        del lr_scheduler
        try:
            log_event(
                "participant_entered_experiment",
                gpu=session.get('gpu_n'),
                incomp_file=session.get('incomp_file'),
                model_file=session.get('model_file'),
            )
        except Exception:
            pass
        ## Present experiment.
        return render_template('experiment.html', workerId=session['workerId'], assignmentId=session['assignmentId'], hitId=session['hitId'], subId=session['subId'], a=session['a'], tp_a=session['tp_a'], b=session['b'], tp_b=session['tp_b'], c=session['c'], tp_c=session['tp_c'])

@bp.route('/experiment', methods=['POST'])
def pass_message():
    """Write jsPsych message to metadata."""
    
    if request.is_json:

        ## Retrieve jsPsych data.
        msg = request.get_json()

        ## Update participant metadata.
        session['MESSAGE'] = msg
        write_metadata(session, ['MESSAGE'], 'a')
        # Log short summary of the message for live monitoring.
        try:
            log_event("experiment_message", message_type=str(type(msg)), keys=list(msg.keys()) if isinstance(msg, dict) else None)
        except Exception:
            pass

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/incomplete_save', methods=['POST'])
def incomplete_save():
    """Save incomplete jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='incomplete')

        try:
            size_hint = len(JSON) if isinstance(JSON, (str, list, dict)) else None
            log_event("incomplete_save", size_hint=size_hint)
        except Exception:
            pass

    ## Flag partial data saving.
    #session['MESSAGE'] = 'incomplete dataset saved'
    #write_metadata(session, ['MESSAGE'], 'a')

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/log_event', methods=['POST'])
def log_event_endpoint():
    """Accept client-side events (e.g. window closed). Throttle: recent activity = tab_hidden only."""
    try:
        data = request.get_json(silent=True) or {}
        event = data.get('event', 'unknown')
        if event == 'window_closed':
            wid = data.get('workerId')
            sid = data.get('subId')
            store = get_status_store()
            ent = store.get(wid, {}) if wid else {}
            last_ts = ent.get('last_seen_ts') or 0
            if (time.time() - last_ts) < 90 and last_ts > 0:
                log_event("participant_tab_hidden", workerId=wid, subId=sid)
            else:
                log_event("participant_window_closed", workerId=wid, subId=sid)
    except Exception:
        pass
    return ('', 200)

@bp.route('/redirect_success', methods = ['POST'])
def redirect_success():
    """Save complete jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='pass')

    ## Flag experiment as complete.
    session['complete'] = 'success'
    write_metadata(session, ['complete'], 'a')
    log_event("participant_exited", status="success")
    try:
        save_participant_experiment_log(session.get('participant_dir') or os.path.join(session['incomplete'].replace('\\\\','\\'), session['subId']), session.get('workerId'), session.get('subId'), 'success')
    except Exception:
        pass

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/redirect_reject', methods = ['POST'])
def redirect_reject():
    """Save rejected jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='reject')

    ## Flag experiment as complete.
    session['complete'] = 'reject'
    write_metadata(session, ['complete'], 'a')
    log_event("participant_exited", status="reject")
    try:
        save_participant_experiment_log(session.get('participant_dir') or os.path.join(session['incomplete'].replace('\\\\','\\'), session['subId']), session.get('workerId'), session.get('subId'), 'reject')
    except Exception:
        pass

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

@bp.route('/redirect_error', methods = ['POST'])
def redirect_error():
    """Save rejected jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='reject')

    ## Flag experiment as complete.
    session['complete'] = 'error'
    write_metadata(session, ['complete'], 'a')
    log_event("participant_exited", status="error")
    try:
        save_participant_experiment_log(session.get('participant_dir') or os.path.join(session['incomplete'].replace('\\\\','\\'), session['subId']), session.get('workerId'), session.get('subId'), 'error')
    except Exception:
        pass

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)

#manager = Manager()
#cur_epoch = manager.Value("d", 0) 

## model function can be defined in experiment.py or other python file
def model_train(dloader, net, optimizer, lr_scheduler, device="cuda"):
    #print("Model_train_In! : {}".format(session["cur_epoch"]))
    for e in range(EPOCHS_PER_TRAIN_ITER):
        acc, loss = iterate_epoch(data_loader=dloader, 
                      network=net, optim=optimizer, eval=False, device=device)
        lr_scheduler.step()
        #print(lr_scheduler.get_lr())
    
    print("E{} || Train ACCs : {:3f}, Train CEs : {:3f}".format(session["cur_epoch"], 
                                                                acc, loss))
    try:
        log_event("model_train_epoch", cur_epoch=session.get("cur_epoch"), acc=float(acc), loss=float(loss), device=device)
    except Exception:
        pass

    session["cur_epoch"] = session["cur_epoch"] + 1
    #print("Model_train_Out : {}".format(session["cur_epoch"]))
   
    return

## function to data from browser and train model one epoch then return model progress
## At end of training, return model result
def _acquire_gpu_lock(device_str, timeout_sec=GPU_LOCK_TIMEOUT):
    """Acquire advisory lock so only one train_iter uses GPU at a time (Unix only). Returns (lock_file, True) or (None, False)."""
    if not _HAS_FCNTL or not (isinstance(device_str, str) and device_str.startswith("cuda:")):
        return None, True
    import tempfile
    lock_path = os.path.join(tempfile.gettempdir(), "online_exp_gpu_train.lock")
    try:
        f = open(lock_path, "w")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # block until we get the lock
        return f, True
    except Exception:
        return None, True  # on error allow training without lock


def _release_gpu_lock(lock_file):
    if lock_file is None:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    except Exception:
        pass


@bp.route('/train_iter', methods = ['GET'])
def train_iter():
    try:
        session['gpu_n'] = _sanitize_device(session.get('gpu_n', 'cpu'))
        log_event("train_iter_start", s_count=session.get("s_count"), cur_epoch=session.get("cur_epoch"), device=session.get("gpu_n"))
        while True:
            try:
                data_np = json_to_nps(os.path.join(session['incomplete'], session['subId'],
                                 '{}_{}.json'.format(session['subId'], session['s_count'])))
                break
            except Exception:
                time.sleep(0.5)
        sessions = list(data_np.keys())
        lock_file, lock_ok = _acquire_gpu_lock(session['gpu_n'])
        if not lock_ok:
            return "0"
        try:
            train_loader, _, _, _, _ = load_os_data([data_np],
                        total_session=np.arange(len(sessions)),
                        train_session=np.arange(len(sessions)),
                        val_session=np.arange(len(sessions)),
                        batch_size=8)
            net = LSTM_Attn2(in_channel=5, dropout=0.0, class_num=2, device=session['gpu_n']);
            lr = 0.01
            optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
            lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                                    cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                                    warmup_steps=150, gamma=0.5)
            net.load_model(fname=session['model_file'].format(session['s_count']), optimizer = optimizer, lr_scheduler=lr_scheduler)
            net = net.to(session['gpu_n'])
            lr_scheduler.set_optimizer(optimizer)
            if not "cur_epoch" in session:
                session["cur_epoch"] = 0
            model_train(train_loader, net, optimizer, lr_scheduler, device=session['gpu_n'])
            net.save_model(fname=session['model_file'].format(session['s_count']), optimizer = optimizer, lr_scheduler=lr_scheduler)
            if session["cur_epoch"] >= total_epoch :
                session["cur_epoch"] = 0
                session['s_count']+=1
                net.save_model(fname=session['model_file'].format(session['s_count']), optimizer = optimizer, lr_scheduler=lr_scheduler)
                try:
                    log_event("train_iter_progress", status="session_finished", s_count=session.get("s_count"), progress=1.0)
                except Exception:
                    pass
                del net
                del optimizer
                return "1"
            else:
                progress = float(session["cur_epoch"]/total_epoch)
                try:
                    log_event("train_iter_progress", status="in_progress", s_count=session.get("s_count"), progress=progress)
                except Exception:
                    pass
                del net
                del optimizer
                return str(progress)
        finally:
            _release_gpu_lock(lock_file)
    except Exception as e:
        traceback.print_exc()
        try:
            log_event("train_iter_error", error=str(e))
        except Exception:
            pass
        return "0"
    
@bp.route('/dummy_iter', methods = ['GET'])
def dummy_iter():
    if not "cur_epoch" in session:
        session["cur_epoch"] = 0
    #print("Dummy train_In! : {}".format(session["cur_epoch"]))
    
    time.sleep(3)
    
    session["cur_epoch"] = session["cur_epoch"] + 1

    #print("Dummy train_Out : {}".format(session["cur_epoch"]))
    
    if session["cur_epoch"] >= total_epoch : 
        info = {"progress" : "1"}
        try:
            log_event("dummy_progress", progress=1.0, cur_epoch=session.get("cur_epoch"), total_epoch=total_epoch)
        except Exception:
            pass
        return json.dumps(info)
    else :
        progress = float(session["cur_epoch"]/total_epoch)
        info = {"progress" : str(progress)}
        try:
            log_event("dummy_progress", progress=progress, cur_epoch=session.get("cur_epoch"), total_epoch=total_epoch)
        except Exception:
            pass
        return json.dumps(info)


@bp.route('/gen_exps', methods = ['GET'])
def gen_exps():
    try:
        session['gpu_n'] = _sanitize_device(session.get('gpu_n', 'cpu'))
        #print("GEN EPISODES!!!!")
        from app.os_model.exp.Oneshot import Oneshot

        os_type = int(request.args.get('ostype'))

        feature_dim=5;
        stim_ratio=[16,8,1];
        stim_vectors = [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                                           np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
                                           np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])]
        cue_ratio = [4,1]
        cue_vectors = [np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                       np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]

        os_exp = Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                    cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.36, recency=0.36, 
                    model="sangwan2014")
        
        net = LSTM_Attn2(in_channel=5, dropout=0.0, class_num=2, device=session['gpu_n']);
        lr = 0.01
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98));
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=200,
                                                     cycle_mult=1.0, max_lr=lr, min_lr=lr * 0.01,
                                                     warmup_steps=150, gamma=0.5)

        net.load_model(fname=session['model_file'].format(session['s_count']),
                       optimizer = optimizer, lr_scheduler = lr_scheduler)
        net = net.to(session['gpu_n'])
        with torch.no_grad() :
            x, _, (cue_alloc, rew_alloc) = os_exp.gen_episode(net, target=os_type, opt_model=os_exp, device=session['gpu_n'])

        info = {"cue_alloc" : cue_alloc.tolist(),
         "rew_alloc" : rew_alloc.tolist()}
        del net
        
        try:
            log_event("gen_exps_success", os_type=os_type)
        except Exception:
            pass
        return json.dumps(info)
    except Exception as e:
        traceback.print_exc()
        try:
            log_event("gen_exps_error", error=str(e))
        except Exception:
            pass
        return json.dumps({"error": True, "cue_alloc": [], "rew_alloc": []})
