import os
from datetime import datetime

def write_metadata(session, keys, mode='w'):
    """Write metadata to disk.

    Parameters
    ----------
    session : flask session
        Current user session.
    keys : list
        Session keys to write to file.
    mode : r | w | a
        Open file mode.
    """

    ## Define timestamp.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ## Write metadata to disk.
    fout = os.path.join(session['metadata'], session['workerId'])
    with open(fout, mode) as f:
        for k in keys:
            f.write(f'{timestamp}\t{k}\t{session[k]}\n')

def write_data(session, json_payload, method='pass'):
    """Write jsPsych output to disk.

    Parameters
    ----------
    session : flask session
        Current user session.
    json_payload : str | object
        Data from jsPsych: JSON string or already-parsed dict/list.
    method : pass | reject
        Designates target folder for data.
    """
    import json as _json
    raw = json_payload if isinstance(json_payload, str) else _json.dumps(json_payload)

    ## Write data to disk.
    if method == 'pass':
        fout = os.path.join(session['data'], '%s.json' % session['subId'])
    elif method == 'reject':
        fout = os.path.join(session['reject'], '%s.json' % session['subId'])
    elif method == 'incomplete':
        fout = os.path.join(session['incomplete'], session['subId'],
                             '{}_{}.json'.format(session['subId'], session['s_count']))
    else:
        return

    with open(fout, 'w', encoding='utf-8') as f:
        f.write(raw)

    ## On successful completion, run diligence analysis and save next to data file.
    if method == 'pass':
        try:
            from app.analysis import run_and_save_analysis
            run_and_save_analysis(session['data'], session['subId'], raw)
        except Exception:
            pass
