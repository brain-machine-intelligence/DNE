/**
 * jspsych-image-slider-response
 * a jspsych plugin for free response survey questions
 *
 * Josh de Leeuw
 *
 * documentation: docs.jspsych.org
 *
 */


jsPsych.plugins['image-slider-response'] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('image-slider-response', 'stimulus', 'image');

  plugin.info = {
    name: 'image-slider-response',
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The image to be displayed'
      },
      stimulus_height: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image height',
        default: null,
        description: 'Set the image height in pixels'
      },
      stimulus_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Image width',
        default: null,
        description: 'Set the image width in pixels'
      },
      maintain_aspect_ratio: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Maintain aspect ratio',
        default: true,
        description: 'Maintain the aspect ratio after setting width or height'
      },
      min: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Min slider',
        default: 0,
        description: 'Sets the minimum value of the slider.'
      },
      max: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Max slider',
        default: 100,
        description: 'Sets the maximum value of the slider',
      },
      slider_start: {
				type: jsPsych.plugins.parameterType.INT,
				pretty_name: 'Slider starting value',
				default: 50,
				description: 'Sets the starting value of the slider',
			},
      step: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Step',
        default: 1,
        description: 'Sets the step of the slider'
      },
      labels: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name:'Labels',
        default: [],
        array: true,
        description: 'Labels of the slider.',
      },
      labels_description: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name:'Labels description',
        default: [],
        array: true,
        description: 'Labels description of the slider. (start, mid, end)',
      },
      color: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'label color',
        default:  'blue',
        array: false,
        description: 'color of label'
      },
      slider_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name:'Slider width',
        default: null,
        description: 'Width of the slider in pixels.'
      },
      button_label: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        array: false,
        description: 'Label of the button to advance.'
      },
      require_movement: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Require movement',
        default: false,
        description: 'If true, the participant will have to move the slider before continuing.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the slider.'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show the trial.'
      },
//// change      
      choice_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'choice duration',
        default: null,
        description: 'How long to wait for a response (in milliseconds).'
      },     
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, trial will end when user makes a response.'
      },
      render_on_canvas: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Render on canvas',
        default: true,
        description: 'If true, the image will be drawn onto a canvas element (prevents blank screen between consecutive images in some browsers).'+
          'If false, the image will be shown via an img element.'
      }
    }
  }

  plugin.trial = function(display_element, trial) {

    var height, width;
    var html;
    // half of the thumb width value from jspsych.css, used to adjust the label positions
    var half_thumb_width = 7.5; 
    
    if (trial.render_on_canvas) {
      var image_drawn = false;
      // first clear the display element (because the render_on_canvas method appends to display_element instead of overwriting it with .innerHTML)
      if (display_element.hasChildNodes()) {
        // can't loop through child list because the list will be modified by .removeChild()
        while (display_element.firstChild) {
          display_element.removeChild(display_element.firstChild);
        }
      }
      // create wrapper div, canvas element and image
      var content_wrapper = document.createElement('div');
      content_wrapper.id = "jspsych-image-slider-response-wrapper";
      content_wrapper.style.margin = "100px 0px";
      var canvas = document.createElement("canvas");
      canvas.id = "jspsych-image-slider-response-stimulus";
      canvas.style.margin = 0;
      canvas.style.padding = 0;
      var ctx = canvas.getContext("2d");
      var img = new Image();   
      img.onload = function() {
        // if image wasn't preloaded, then it will need to be drawn whenever it finishes loading
        if (!image_drawn) {
          getHeightWidth(); // only possible to get width/height after image loads
          ctx.drawImage(img,0,0,width,height);
        }
      };
      img.src = trial.stimulus;
      // get/set image height and width - this can only be done after image loads because uses image's naturalWidth/naturalHeight properties
      function getHeightWidth() {
        if (trial.stimulus_height !== null) {
          height = trial.stimulus_height;
          if (trial.stimulus_width == null && trial.maintain_aspect_ratio) {
            width = img.naturalWidth * (trial.stimulus_height/img.naturalHeight);
          }
        } else {
          height = img.naturalHeight;
        }
        if (trial.stimulus_width !== null) {
          width = trial.stimulus_width;
          if (trial.stimulus_height == null && trial.maintain_aspect_ratio) {
            height = img.naturalHeight * (trial.stimulus_width/img.naturalWidth);
          }
        } else if (!(trial.stimulus_height !== null & trial.maintain_aspect_ratio)) {
          // if stimulus width is null, only use the image's natural width if the width value wasn't set 
          // in the if statement above, based on a specified height and maintain_aspect_ratio = true
          width = img.naturalWidth;
        }
        canvas.height = height;
        canvas.width = width;
      }
      getHeightWidth(); // call now, in case image loads immediately (is cached)
      // create container with slider and labels
      var slider_container = document.createElement('div');
      slider_container.classList.add("jspsych-image-slider-response-container");
      slider_container.style.position = "relative";
      slider_container.style.margin = "0 auto 3em auto";
      if(trial.slider_width !== null){
        slider_container.style.width = trial.slider_width.toString()+'px';
      }
      // create html string with slider and labels, and add to slider container
      html ='<input type="range" class="jspsych-slider" value="'+trial.slider_start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" id="jspsych-image-slider-response-response" ></input>';
      html += '<div>'
      for(var j=0; j < trial.labels.length; j++){
        var label_width_perc = 100/(trial.labels.length-1);
        var percent_of_range = j * (100/(trial.labels.length - 1));
        var percent_dist_from_center = ((percent_of_range-50)/50)*100;
        var offset = (percent_dist_from_center * half_thumb_width)/100;
        html += '<div style="border: 1px solid transparent; display: inline-block; position: absolute; '+
        'left:calc('+percent_of_range+'% - ('+label_width_perc+'% / 2) - '+offset+'px); text-align: center; width: '+label_width_perc+'%;">';
        html += '<span style="text-align: center; font-size: 80%;">'+trial.labels[j]+'</span>';
        html += '</div>'
      }
      html += '</div>';
      slider_container.innerHTML = html;
      // add canvas and slider to content wrapper div
      content_wrapper.insertBefore(canvas, content_wrapper.firstElementChild);
      content_wrapper.insertBefore(slider_container, canvas.nextElementSibling);
      content_wrapper.autofocus=true
      // add content wrapper div to screen and draw image on canvas
      display_element.insertBefore(content_wrapper, null);
      if (img.complete && Number.isFinite(width) && Number.isFinite(height)) {
        // if image has loaded and width/height have been set, then draw it now
        // (don't rely on img onload function to draw image when image is in the cache, because that causes a delay in the image presentation)
        ctx.drawImage(img,0,0,width,height);
        image_drawn = true;  
      }
      // add prompt if there is one
      if (trial.prompt !== null) {
        display_element.insertAdjacentHTML('beforeend', trial.prompt);
      }
      // add submit button
      
      /*
      var submit_btn = document.createElement('button');
      submit_btn.id = "jspsych-image-slider-response-next";
      submit_btn.classList.add("jspsych-btn");
      submit_btn.disabled = (trial.require_movement) ? true : false;
      submit_btn.innerHTML = trial.button_label;
      display_element.insertBefore(submit_btn, display_element.nextElementSibling);
      */
    } else {

      html = '';

      if (trial.prompt !== null){
        html += trial.prompt;
      }

      html += '<div id="jspsych-image-slider-response-wrapper" style="margin: 100px 0px;">';
      html += '<div id="jspsych-image-slider-response-stimulus">';
      html += '<img src="'+trial.stimulus+'" style="';
      if(trial.stimulus_height !== null){
        html += 'height:'+trial.stimulus_height+'px; '
        if(trial.stimulus_width == null && trial.maintain_aspect_ratio){
          html += 'width: auto; ';
         }
      }
      if(trial.stimulus_width !== null){
        html += 'width:'+trial.stimulus_width+'px; '
        if(trial.stimulus_height == null && trial.maintain_aspect_ratio){
          html += 'height: auto; ';
        }
      }
      html += '"></img>';
      html += '</div>';
      html += '<div class="jspsych-image-slider-response-container" style="position:relative; margin: 0 auto 3em auto; width:';
      if (trial.slider_width !== null) {
        html += trial.slider_width+'px;';
      } else {
        html += 'auto;';
      }
      html += '">';
      html += '<input type="range" class="jspsych-slider" value="'+trial.slider_start+'" min="'+trial.min+'" max="'+trial.max+'" step="'+trial.step+'" id="jspsych-image-slider-response-response"></input>';
      html += '<div>'
      for(var j=0; j < trial.labels.length; j++){
        var label_width_perc = 100/(trial.labels.length-1);
        var percent_of_range = j * (100/(trial.labels.length - 1));
        var percent_dist_from_center = ((percent_of_range-50)/50)*100;
        var offset = (percent_dist_from_center * half_thumb_width)/100;
        html += '<div style="border: 1px solid transparent; display: inline-block; position: absolute; '+
        'left:calc('+percent_of_range+'% - ('+label_width_perc+'% / 2) - '+offset+'px); text-align: center; width: '+label_width_perc+'%;">';
        
        if (j==0) {
          html += '<div style="color:'+trial.color+';text-align: center; font-size: 80%;">'+trial.labels[j]+'<br>'+trial.labels_description[0]+'</div>';
        } else if (j==parseInt((trial.labels.length-1)/2)){
          html += '<div style="color:'+trial.color+';text-align: center; font-size: 80%;">'+trial.labels[j]+'<br>'+trial.labels_description[1]+'</div>';
        } else if (j==(trial.labels.length-1)){
          html += '<div style="color:'+trial.color+';text-align: center; font-size: 80%;">'+trial.labels[j]+'<br>'+trial.labels_description[2]+'</div>';
        }  else {
          html += '<div style="color:'+trial.color+';text-align: center; font-size: 80%;">'+trial.labels[j]+'</div>';
        }
        
        html += '</div>'
      }

      html += '</div>';
      html += '</div>';
      html += '</div>';     

      display_element.innerHTML = html;
      
      
      // set image dimensions after image has loaded (so that we have access to naturalHeight/naturalWidth)
      var img = display_element.querySelector('img');
      if (trial.stimulus_height !== null) {
        height = trial.stimulus_height;
        if (trial.stimulus_width == null && trial.maintain_aspect_ratio) {
          width = img.naturalWidth * (trial.stimulus_height/img.naturalHeight);
        }
      } else {
        height = img.naturalHeight;
      }
      if (trial.stimulus_width !== null) {
        width = trial.stimulus_width;
        if (trial.stimulus_height == null && trial.maintain_aspect_ratio) {
          height = img.naturalHeight * (trial.stimulus_width/img.naturalWidth);
        }
      } else if (!(trial.stimulus_height !== null & trial.maintain_aspect_ratio)) {
        // if stimulus width is null, only use the image's natural width if the width value wasn't set 
        // in the if statement above, based on a specified height and maintain_aspect_ratio = true
        width = img.naturalWidth;
      }
      img.style.height = height.toString() + "px";
      img.style.width = width.toString() + "px";
      
    }

    var response = {
      rt: null,
      time_range: null,
      msg_time_range: null,
      response: null,
      description: null,
    };
    
    // set focus to input slider
    var refocus = function(){
      if (display_element.querySelector('#jspsych-image-slider-response-response') != null){
        display_element.querySelector('#jspsych-image-slider-response-response').focus();
      }
    }

    //refocus();
    display_element.querySelector('#jspsych-image-slider-response-response').focus();
    
    display_element.querySelector('#jspsych-image-slider-response-response').addEventListener('change', createTimeout2);
    
    document.addEventListener('click',refocus);

    // Add keydown event listener to disable other keys except left/right arrow key
    var slider = document.getElementById('jspsych-image-slider-response-response');
    slider.onkeydown = function(e){
      allowlist = ['ArrowRight','ArrowLeft']
      if(!(allowlist.includes(e.key))){ //allow keys: arrowright, arrowleft
        //console.log(e.key)
        e.preventDefault()
      }
    }


    var one_msg_overtime = msg_overtime;
    var startTime = performance.now();
    now_start_time = startTime;
    //console.log('real start'+ startTime);
    var msg_startTime = null;
    var msg_endTime = null;
    var remain_timeout2_time = null;
    var rating_gap_duration = null;

//// change to keypress, missed_response

    // timeout for submit time limit 
    var createTimeout1 = function(msg_time) {
      if (trial.trial_duration !== null) {
        jsPsych.pluginAPI.setTimeout(function() {
          missed_response(true);
        }, (trial.trial_duration-msg_time));
      }  
    }

    var end_trial = function(gap){

      //console.log("slider end")
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      //clearTimeout(Timeout2);
      //(document.body).removeEventListener('click',refocus);
      
      msg_overtime = one_msg_overtime;

      // save data
      path = (trial.stimulus).split("/");

      var trialdata = {
        session: parseInt(repetition_count/8),
        round:(repetition_count%8),
        question: [path.pop(), response.response],
        question_timeline: response.time_range,
        msg_timeline: response.msg_time_range,
        description: response.description,
      };
      
      if (gap) {
        html_gap = "<p style='font-familiy:IBM Plex Sans Extrabold;font-size:24px;'>+</p>";
        display_element.innerHTML = html_gap;
        
        rating_gap_duration = Math.floor(Math.random()*(100)+200);

        clearTimeout(Timeout2);
        //console.log("timeout2_start "+timeout2_start)
        remain_timeout2_duration = timeout_duration-((performance.now())-timeout2_start);
        //console.log("remain timeout2 " + remain_timeout2_duration);

        jsPsych.pluginAPI.setTimeout(function() {
          display_element.innerHTML = "";
          Timeout2 = window.setTimeout(timeout_response, remain_timeout2_duration); 
          timeout2_start = timeout2_start + rating_gap_duration;
        //  console.log("plus timeout2_start "+timeout2_start)
          jsPsych.finishTrial(trialdata);
        }, (rating_gap_duration));

      } else {
        display_element.innerHTML = "";
        jsPsych.finishTrial(trialdata); 
      }

      }

    // no keyboard response = no submit
    var missed_response = function(end_gap){
 
      // Kill all setTimeout handlers.
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      display_element.querySelector('#jspsych-image-slider-response-response').removeEventListener('change', createTimeout2);
         
      document.removeEventListener('click',refocus); 

      var endTime = performance.now();
      response.rt = endTime - now_start_time;
      response.time_range = [now_start_time, endTime];
      response.msg_time_range = [msg_startTime, msg_endTime];
      response.response = display_element.querySelector('#jspsych-image-slider-response-response').valueAsNumber;
      response.description = "no submit"
      
      end_trial(end_gap);
    }

    // show msg over two rating
    var over_timeout_response = function() {

      console.log("over")

      // Kill all setTimeout handlers.
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      clearTimeout(Timeout2);
      display_element.querySelector('#jspsych-image-slider-response-response').removeEventListener('change', createTimeout2);
      
      document.removeEventListener('click',refocus);

      // Display warning message.
      if (msg_count < chance) {
        var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">You did not respond within the allotted time. Please pay more attention on the next trial.<br><br><b>Warning:</b> If you miss too many trials, we may end the experiment early and reject your work.';
      }
      else {
        var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">we end the experiment early and reject your work</p>';
      }

      display_element.innerHTML = msg;


      one_msg_overtime = 0;

      if (msg_count >= chance) {
        low_quality = true;
        msg_timeout = window.setTimeout(function() {
          jsPsych.endExperiment();
         }, msg_duration); 
      } else {
        msg_timeout = window.setTimeout(function() {
          msg_endTime = performance.now();
          html_gap = "<p style='font-familiy:IBM Plex Sans Extrabold;font-size:24px;'>+</p>";
          display_element.innerHTML = html_gap;
          rating_gap_duration = Math.floor(Math.random()*(100)+200);
          
          jsPsych.pluginAPI.setTimeout(function() {
            display_element.innerHTML = html;
            display_element.querySelector('#jspsych-image-slider-response-response').focus();
            display_element.querySelector('#jspsych-image-slider-response-response').addEventListener('change', createTimeout2);
             
            document.addEventListener('click',refocus);
            createTimeout2();
            createkeyboardListener();
            createTimeout1(msg_overtime);
            clearTimeout(msg_timeout);
          }, (rating_gap_duration));


          }, msg_overtime);  
      }
    }

    // no response past the time limit  
    var timeout_response = function() {

      console.log("timeout")

      // Kill all setTimeout handlers.
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      clearTimeout(Timeout2);
      display_element.querySelector('#jspsych-image-slider-response-response').removeEventListener('change', createTimeout2);
      document.removeEventListener('click',refocus);

      msg_count += 1; 

      // Display warning message.
      if (msg_count < chance) {
        var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">You did not respond within the allotted time. Please pay more attention on the next trial.<br><br><b>Warning:</b> If you miss too many trials, we may end the experiment early and reject your work.';
      }
      else {
        var msg = '<p id = "msg" style="font-size: 20px; line-height: 1.5em">we end the experiment early and reject your work</p>';
      }

      display_element.innerHTML = msg;
      
      var curr_time = performance.now();
      msg_startTime = performance.now();
      var remain_time = (trial.trial_duration - (msg_startTime-now_start_time)); 

      //console.log("real start " + now_start_time );
      //console.log("msg start " + msg_startTime );
      //console.log("remain " + remain_time );
      //console.log("timeout1 "+(remain_time-msg_duration))
    
        if (remain_time > msg_duration){ 
          if (msg_count >= chance) {
            low_quality = true;
            msg_timeout = window.setTimeout(function() {
              jsPsych.endExperiment();
             }, msg_duration); 
          } else {
            one_msg_overtime = 0;
            msg_timeout = window.setTimeout(function() {
              msg_endTime = performance.now();
              display_element.innerHTML = html;
              display_element.querySelector('#jspsych-image-slider-response-response').focus();
              display_element.querySelector('#jspsych-image-slider-response-response').addEventListener('change', createTimeout2);
               
              document.addEventListener('click',refocus);
              createTimeout2();
              createkeyboardListener();
              createTimeout1(trial.trial_duration-(remain_time-msg_duration));
              clearTimeout(msg_timeout);
             }, msg_duration); 
          }
        } else {
          one_msg_overtime = msg_duration - remain_time;
          //console.log("over " + one_msg_overtime );
          //console.log("remain "+remain_time)
          msg_timeout = window.setTimeout(function() {
            display_element.innerHTML = html;
            display_element.querySelector('#jspsych-image-slider-response-response').focus();
            display_element.querySelector('#jspsych-image-slider-response-response').addEventListener('change', createTimeout2);
            
            document.addEventListener('click',refocus);
            createkeyboardListener();
            clearTimeout(msg_timeout);
            missed_response(false);
          }, remain_time); 
        }
    }
     
    // callback function of keyboardlistener ('y' press)
    var after_response = function(){
 
      // Kill all setTimeout handlers.
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      display_element.querySelector('#jspsych-image-slider-response-response').removeEventListener('change', createTimeout2);
       
      document.removeEventListener('click',refocus);  
      
      createTimeout2();
      
      var endTime = performance.now();
      response.rt = endTime - now_start_time;
      response.time_range = [now_start_time, endTime];
      response.msg_time_range = [msg_startTime, msg_endTime];
      response.response = display_element.querySelector('#jspsych-image-slider-response-response').valueAsNumber;

      if (response.rt <= trial.choice_duration){
        response.description = "correct submit"
      } else {
        response.description = "late submit"
      }
      end_trial(true);
      
    }
 
    var createkeyboardListener = function() {
      var keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: ['y'],
        rt_method: 'performance',
        persist: false,
      });  
    } 
    

    createkeyboardListener();
    if (msg_overtime == 0){
      createTimeout1(msg_overtime);
    } else {
      over_timeout_response();
    }

    // timeout for response time limit
    if (Timeout2 == null) {
      Timeout2 = window.setTimeout(timeout_response, timeout_duration); 
      timeout2_start = performance.now();
    }

    function createTimeout2() {
      clearTimeout(Timeout2);
      Timeout2 = window.setTimeout(timeout_response, timeout_duration);
      timeout2_start = performance.now();
      //console.log("timer reset " + timeout2_start );
    }


  };

  return plugin;
})();
