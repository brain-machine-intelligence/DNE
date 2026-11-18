import os, sys, re, configparser, warnings, logging
from flask import (Flask, redirect, render_template, request, session, url_for)
from app import consent, alert, experiment, complete, error
from .io import write_metadata
from .utils import gen_code
from .dne_log import log_event, get_status_store, get_recent_errors, get_recent_log, update_heartbeat
from werkzeug.middleware.proxy_fix import ProxyFix
__version__ = '1.2'

## Define root directory.
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

## Load and parse configuration file.
cfg = configparser.ConfigParser()
cfg.read(os.path.join(ROOT_DIR, 'app.ini'))

## Ensure output directories exist.
data_dir = os.path.join(ROOT_DIR, cfg['IO']['DATA'])
if not os.path.isdir(data_dir): os.makedirs(data_dir)
meta_dir = os.path.join(ROOT_DIR, cfg['IO']['METADATA'])
if not os.path.isdir(meta_dir): os.makedirs(meta_dir)
incomplete_dir = os.path.join(ROOT_DIR, cfg['IO']['INCOMPLETE'])
if not os.path.isdir(incomplete_dir): os.makedirs(incomplete_dir)
reject_dir = os.path.join(ROOT_DIR, cfg['IO']['REJECT'])
if not os.path.isdir(reject_dir): os.makedirs(reject_dir)

## Check Flask mode; if debug mode, clear session variable.
debug = cfg['FLASK'].getboolean('DEBUG')
if debug:
    warnings.warn("WARNING: Flask currently in debug mode. This should be changed prior to production.")

## Check Flask password.
secret_key = cfg['FLASK']['SECRET_KEY']
if secret_key == "PLEASE_CHANGE_THIS":
    warnings.warn("WARNING: Flask password is currently default. This should be changed prior to production.")

## Check restart mode; if true, participants can restart experiment.
allow_restart = cfg['FLASK'].getboolean('ALLOW_RESTART')

## Initialize Flask application.
app = Flask(__name__)
app.secret_key = secret_key

app.config['APPLICATION_ROOT'] = '/dne/api'

## Logging: one line per event so journalctl/console shows [DNE] events clearly.
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    app.logger.addHandler(_handler)

app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_proto=1,
    x_host=1,
    x_prefix=1
)

## Apply blueprints to the application.
app.register_blueprint(consent.bp)
app.register_blueprint(alert.bp)
app.register_blueprint(experiment.bp)
app.register_blueprint(complete.bp)
app.register_blueprint(error.bp)


@app.route('/ping')
def ping():
    """Heartbeat: update last_seen and optional progress; log when progress changes."""
    try:
        if session.get('workerId'):
            kw = {}
            for k in ('session', 'round', 'repetition', 'total'):
                v = request.args.get(k)
                if v is not None:
                    try:
                        kw[k] = int(v)
                    except ValueError:
                        kw[k] = v
            store_before = get_status_store()
            prev = store_before.get(session.get('workerId'), {})
            update_heartbeat(session.get('workerId'), session.get('subId'), **kw)
            if kw:
                store_after = get_status_store()
                cur = store_after.get(session.get('workerId'), {})
                if cur.get('session') != prev.get('session') or cur.get('repetition') != prev.get('repetition'):
                    log_event("progress", session=cur.get('session'), round=cur.get('round'), repetition=cur.get('repetition'), total=cur.get('total'))
    except Exception:
        pass
    return ('', 200)


def _status_view():
    """Live status table: ID | current state | session/progress | stuck? + recent errors + recent log."""
    from datetime import datetime
    import time
    store = get_status_store()
    now = time.time()
    stuck_sec = 300
    rows = []
    for wid, data in sorted(store.items()):
        sub_id = data.get('subId', '')[:12]
        last_ev = (data.get('last_event') or '')[:28]
        ts = data.get('last_seen_ts') or 0
        last_seen = datetime.fromtimestamp(ts).strftime('%H:%M:%S') if ts else '-'
        stuck = 'Y' if (now - ts) > stuck_sec else ''
        s_count = data.get('s_count')
        session_num = data.get('session')
        progress = data.get('progress')
        cur_epoch = data.get('cur_epoch')
        total_epoch = data.get('total_epoch')
        repetition = data.get('repetition')
        total = data.get('total')
        if session_num is not None and total is not None:
            prog_str = "s%d r%d/%d" % (session_num, (repetition or 0) % 8, 8)
        elif s_count is not None and progress is not None:
            prog_str = "s%d %.0f%%" % (s_count, float(progress) * 100)
        elif cur_epoch is not None and total_epoch is not None:
            prog_str = "e%d/%d" % (cur_epoch, total_epoch)
        elif progress is not None:
            prog_str = "%.0f%%" % (float(progress) * 100)
        else:
            prog_str = '-'
        rows.append((wid[:14], sub_id, last_ev, prog_str, last_seen, stuck))
    error_entries = get_recent_errors()
    error_rows = [(e["ts"], e["event"], e["workerId"], e["subId"], e["message"]) for e in reversed(error_entries)]
    recent_log_entries = get_recent_log()
    recent_log_rows = [(e["ts"], e["event"], e["workerId"], e["subId"], e["message"]) for e in reversed(recent_log_entries)]
    return render_template('status.html', rows=rows, error_rows=error_rows, recent_log_rows=recent_log_rows)


@app.route('/status')
def status():
    return _status_view()


@app.route('/dne/api/status')
def status_prefixed():
    """Same as /status, for when the app is served at root and URL is /dne/api/status."""
    return _status_view()


## Define root node.
@app.route('/')
def index():

    ## Debug mode: clear session.
    if debug:
        session.clear()

    ## Store directories in session object.
    session['data'] = data_dir
    session['metadata'] = meta_dir
    session['incomplete'] = incomplete_dir
    session['reject'] = reject_dir
    session['allow_restart'] = allow_restart

    ## Record incoming metadata.
    info = dict(
        workerId     = request.args.get('workerId'),        # MTurk metadata
        assignmentId = request.args.get('assignmentId'),    # MTurk metadata
        hitId        = request.args.get('hitId'),           # MTurk metadata
        subId        = gen_code(24, W_ID=request.args.get('workerId')), # NivTurk metadata
        a            = request.args.get('a'),               # TurkPrime metadata
        tp_a         = request.args.get('tp_a'),            # TurkPrime metadata
        b            = request.args.get('b'),               # TurkPrime metadata
        tp_b         = request.args.get('tp_b'),            # TurkPrime metadata
        c            = request.args.get('c'),               # TurkPrime metadata
        tp_c         = request.args.get('tp_c'),            # TurkPrime metadata
        address      = request.remote_addr,                 # NivTurk metadata
        browser      = request.user_agent.browser,          # User metadata
        platform     = request.user_agent.platform,         # User metadata
        version      = request.user_agent.version,          # User metadata
    )

    ## Case 1: workerId absent form URL.
    if info['workerId'] is None:
        try:
            log_event("participant_rejected", reason="no_workerId")
        except Exception:
            pass
        ## Redirect participant to error (missing workerId).
        return redirect(url_for('error.error', errornum=1000))

    ## Case 2: mobile / tablet user.
    elif info['platform'] in ['android','iphone','ipad','wii']:
        try:
            log_event("participant_rejected", reason="mobile_platform", workerId=info.get('workerId'))
        except Exception:
            pass
        ## Redirect participant to error (platform error).
        return redirect(url_for('error.error', errornum=1001))

    ## Case 3: previous complete.
    elif 'complete' in session:

        ## Redirect participant to complete page.
        return redirect(url_for('complete.complete'))

    ## Case 4: repeat visit, manually changed workerId.
    elif 'workerId' in session and session['workerId'] != info['workerId']:

        ## Update metadata.
        session['ERROR'] = '1005: workerId tampering detected.'
        session['complete'] = 'error'
        write_metadata(session, ['ERROR','complete'], 'a')
        try:
            log_event("participant_exited", status="error", reason="workerId_tampering")
        except Exception:
            pass

        ## Redirect participant to error (unusual activity).
        return redirect(url_for('error.error', errornum=1005))

    ## Case 5: repeat visit, preexisting activity.
    elif 'workerId' in session:

        ## Redirect participant to consent form.
        return redirect(url_for('consent.consent'))

    ## Case 6: repeat visit, preexisting log but no session data.
    elif not 'workerId' in session and info['workerId'] in os.listdir(meta_dir):

        ## Parse log file.
        with open(os.path.join(session['metadata'], info['workerId']), 'r') as f:
            logs = f.read()

        ## Extract subject ID.
        info['subId'] = re.search('subId\t(.*)\n', logs).group(1)

        ## Check for previous consent.
        consent = re.search('consent\t(.*)\n', logs)
        if consent and consent.group(1) == 'True': info['consent'] = True       # consent = true
        elif consent and consent.group(1) == 'False': info['consent'] = False   # consent = false
        elif consent: info['consent'] = consent.group(1)                        # consent = bot

        ## Check for previous experiment.
        experiment = re.search('experiment\t(.*)\n', logs)
        if experiment: info['experiment'] = experiment.group(1)

        ## Check for previous complete.
        complete = re.search('complete\t(.*)\n', logs)
        if complete: info['complete'] = complete.group(1)

        ## Update metadata.
        for k, v in info.items(): session[k] = v

        ## Redirect participant as appropriate.
        if 'complete' in session:
            return redirect(url_for('complete.complete'))
        elif 'experiment' in session:
            return redirect(url_for('experiment.experiment'))
        else:
            return redirect(url_for('consent.consent'))

    ## Case 7: first visit, workerId present.
    else:

        ## Update metadata.
        for k, v in info.items(): session[k] = v
        write_metadata(session, ['workerId','hitId','assignmentId','subId','address','browser','platform','version'], 'w')
        try:
            log_event("participant_entered", workerId=info['workerId'], subId=info['subId'], address=info.get('address'))
        except Exception:
            pass

        ## Redirect participant to consent form.
        return redirect(url_for('consent.consent'))
