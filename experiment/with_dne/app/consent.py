from flask import (Blueprint, redirect, render_template, request, session, url_for)
from .io import write_metadata
from .dne_log import log_event

## Initialize blueprint.
bp = Blueprint('consent', __name__)

@bp.route('/consent')
def consent():
    """Present consent form to participant."""

    ## Error-catching: screen for missing session.
    if not 'workerId' in session:
        try:
            log_event("participant_rejected", reason="no_workerId_consent")
        except Exception:
            pass
        ## Redirect participant to error (missing workerId).
        return redirect(url_for('error.error', errornum=1000))

    ## Case 1: previously completed experiment.
    elif 'complete' in session:

        ## Redirect participant to complete page.
        return redirect(url_for('complete.complete'))

    ## Case 2: first visit.
    elif not 'consent' in session:
        try:
            log_event("participant_consent_page")
        except Exception:
            pass
        ## Present consent form.
        return render_template('consent.html', workerId=session.get('workerId'), subId=session.get('subId'))

    ## Case 3: repeat visit, previous bot-detection.
    elif session['consent'] == 'BOT':

        ## Redirect participant to error (unusual activity).
        return redirect(url_for('error.error', errornum=1005))

    ## Case 4: repeat visit, previous non-consent.
    elif session['consent'] == False:

        ## Redirect participant to error (decline consent).
        return redirect(url_for('error.error', errornum=1002))

    ## Case 5: repeat visit, previous consent.
    else:

        ## Redirect participant to alert page.
        return redirect(url_for('alert.alert'))

@bp.route('/consent', methods=['POST'])
def consent_post():
    """Process participant repsonse to consent form."""

    ## Retrieve participant response.
    subj_consent = int(request.form['subj_consent'])
    bot_check = request.form.get('future_contact', False)

    ## Check for suspicious responding.
    if bot_check:

        ## Update participant metadata.
        session['consent'] = 'BOT'
        session['complete'] = 'error'
        write_metadata(session, ['consent','complete'], 'a')
        try:
            log_event("participant_exited", status="error", reason="bot")
        except Exception:
            pass
        ## Redirect participant to error (unusual activity).
        return redirect(url_for('error.error', errornum=1005))

    ## Check participant response.
    elif subj_consent:

        ## Update participant metadata.
        session['consent'] = True
        write_metadata(session, ['consent'], 'a')
        try:
            log_event("participant_consent_accepted")
        except Exception:
            pass
        ## Redirect participant to alert page.
        return redirect(url_for('alert.alert'))

    else:

        ## Update participant metadata.
        session['consent'] = False
        session['complete'] = 'error'
        write_metadata(session, ['consent'], 'a')
        try:
            log_event("participant_exited", status="error", reason="consent_declined")
        except Exception:
            pass
        ## Redirect participant to error (decline consent).
        return redirect(url_for('error.error', errornum=1002))
