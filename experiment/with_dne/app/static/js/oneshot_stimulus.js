// API base 안전하게 확보
var API_BASE = (typeof window !== 'undefined' && window.API_BASE !== undefined)
  ? window.API_BASE
  : '';

// specify the images corresponding to simple cues
// STATIC_BASE is intended to be set by experiment.html via:
//   <script>window.STATIC_BASE = "{{ url_for('static', filename='') }}";</script>
// This file provides a robust fallback if that injection fails.

(function () {
  // 1) Resolve a safe base for static assets
  // Priority:
  //   (A) window.STATIC_BASE from Flask template (recommended): "/dne/static/" or "/dne/api/static/"
  //   (B) fallback derived from current URL: "<origin>/dne/static/" (works if nginx aliases /dne/static/)
  function resolveStaticBase() {
    const injected = (typeof window !== 'undefined') ? window.STATIC_BASE : undefined;

    if (typeof injected === 'string' && injected.trim().length > 0) {
      return injected.replace(/\/?$/, '') + '/';
    }

    // Fallback: attempt to infer from current location.
    // If participant URL is https://host/dne/... -> static assumed at https://host/dne/static/
    // If you serve static elsewhere, adjust nginx or inject window.STATIC_BASE properly.
    if (typeof window !== 'undefined' && window.location) {
      const origin = window.location.origin;
      const path = window.location.pathname || '/';

      // Grab first segment, usually "/dne/..."
      const m = path.match(/^\/([^\/]+)\//);
      const prefix = m ? `/${m[1]}` : '';
      return `${origin}${prefix}/static/`.replace(/\/?$/, '') + '/';
    }

    // Last resort (should not happen in browsers)
    return '/static/';
  }

  const BASE = resolveStaticBase();
  console.log('STATIC_BASE (resolved) =', BASE);

  // expose if other scripts want it
  window.STATIC_BASE_RESOLVED = BASE;

  // 2) Build image lists
  const cue_images = [];
  for (let idx = 1; idx <= 120; idx++) {
    const i_path = BASE + "img/cue_imgs/" + ("00" + idx).slice(-3) + ".png";
    cue_images.push(i_path);
  }

  const tuto_images = [];
  for (let idx = 1; idx <= 3; idx++) {
    const i_path = BASE + "img/cue_imgs/test0" + idx + ".png";
    tuto_images.push(i_path);
  }

  const reward_images_name = ['-10', '-50', '10', '50'];

  const reward_images = [
    BASE + 'img/reward_imgs/msg_n10pt.PNG',
    BASE + 'img/reward_imgs/msg_n50pt.PNG',
    BASE + 'img/reward_imgs/msg_p10pt.PNG',
    BASE + 'img/reward_imgs/msg_p50pt.PNG',
  ];

  const inst_images = [];
  for (let idx = 1; idx <= 37; idx++) {
    const inst_path = BASE + "img/Inst_imgs/inst" + idx + ".PNG";
    inst_images.push(inst_path);
  }

  // 3) Export to global (keeps compatibility with your existing code)
  window.cue_images = cue_images;
  window.tuto_images = tuto_images;
  window.reward_images_name = reward_images_name;
  window.reward_images = reward_images;
  window.inst_images = inst_images;

  // ---- original logic below (minimal changes) ----

  function alloc_cues() {
    // kept as-is; consider initialising if needed elsewhere
    cue_infos = {};
  }

  function alloc_rewards() {
    reward_infos = {};
  }

  // Use the generated cue_images from window
  window.unused_cue_idxes = [...Array(cue_images.length).keys()];


  function set_rounds_type() {
    rounds_type = {};

    // os_type: 0 = oneshot, 1 = incremental
    // n_or_p : 0 = n = (non-novel = +10, novel = -50), 1 = p = (non-novel = -10, novel = +50)

    os_type = [0, 0, 0, 0, 1, 1, 1, 1];
    n_or_p = [0, 0, 0, 0, 1, 1, 1, 1];

    os_type = jsPsych.randomization.shuffle(os_type);
    n_or_p = jsPsych.randomization.shuffle(n_or_p);

    rounds_type.os_type = os_type;
    rounds_type.n_or_p = n_or_p;

    return rounds_type;
  }

  function alloc_all(os_type = 0, n_or_p = 0, driving = 0, verbose = false) { // select images and rewards for one trial
    total_infos = {};

    if (n_or_p == 0) { // n
      total_infos.rew_idxes = [2, 1]; // [non-novel, novel]
      novel_outcome = 1;
      non_novel_outcome = 2;
    } else { // p
      total_infos.rew_idxes = [0, 3];
      novel_outcome = 3;
      non_novel_outcome = 0;
    }

    cue_idxes = jsPsych.randomization.sampleWithoutReplacement(unused_cue_idxes, 3); // randomly sample 3 images
    total_infos.cue_idxes = cue_idxes;

    if (driving == 0) {
      // allocate frequency in which images are presented
      cue_allocation = [...Array(25).keys()];
      cue_allocation = cue_allocation.fill(cue_idxes[0], 0, 16);
      cue_allocation = cue_allocation.fill(cue_idxes[1], 16, 24);
      cue_allocation = cue_allocation.fill(cue_idxes[2], 24);
      cue_allocation = jsPsych.randomization.shuffle(cue_allocation); // shuffle images order

      total_infos.cue_allocation = cue_allocation;

      if (os_type == 0) { // oneshot
        reward_allocation = Array(5).fill(non_novel_outcome);
        reward_allocation[parseInt(cue_allocation.indexOf(cue_idxes[2]) / 5)] = novel_outcome;
      } else { // incremental
        reward_allocation = Array(5).fill(non_novel_outcome);
        novel_cue_idx = parseInt(cue_allocation.indexOf(cue_idxes[2]) / 5);
        nonnovel_cue_idxs = [0, 1, 2, 3, 4];
        nonnovel_cue_idxs.splice(nonnovel_cue_idxs.indexOf(novel_cue_idx), 1);

        novel_reward_idx = jsPsych.randomization.sampleWithoutReplacement(nonnovel_cue_idxs, 1);
        reward_allocation[novel_reward_idx] = novel_outcome;
      }

      total_infos.reward_allocation = reward_allocation;
    } else {
      $.ajax({
        type: "GET",
        url: (window.API_BASE || "/dne/api") + "/gen_exps?ostype=" + os_type + "&norp=" + n_or_p,
        async: false,
        success: function (response) {
          obj = JSON.parse(response);
          if (obj && obj.error) {
            total_infos.error = true;
            total_infos.cue_allocation = [];
            total_infos.reward_allocation = [];
            return;
          }
          cue_allocs = obj.cue_alloc || [];
          rew_allocs = obj.rew_alloc || [];

          for (let i = 0; i < cue_allocs.length; i++) {
            if (cue_allocs[i] == "0") {
              cue_allocs[i] = cue_idxes[0];
            } else if (cue_allocs[i] == "1") {
              cue_allocs[i] = cue_idxes[1];
            } else if (cue_allocs[i] == "2") {
              cue_allocs[i] = cue_idxes[2];
            }
          }

          // Backend returns reward image indices 0,1,2,3 (same as driving=0)
          total_infos.cue_allocation = cue_allocs;
          total_infos.reward_allocation = rew_allocs.map(function (x) { return Number(x); });
        }
      }).fail(function () {
        total_infos.error = true;
        total_infos.cue_allocation = [];
        total_infos.reward_allocation = [];
      });
    }

    return total_infos;
  }

  // Export functions to global for compatibility
  window.alloc_cues = alloc_cues;
  window.alloc_rewards = alloc_rewards;
  window.set_rounds_type = set_rounds_type;
  window.alloc_all = alloc_all;

})();
