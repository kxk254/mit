var deepqlearn = deepqlearn || { REVISION: 'ALPHA' };

(function(global) {
  "use strict";
  
  // An agent is in state0 and does action0
  // environment then assigns reward0 and provides new state, state1
  // Experience nodes store all this information, which is used in the
  // Q-learning update step
  var Experience = function(state0, action0, reward0, state1) {
    this.state0 = state0;
    this.action0 = action0;
    this.reward0 = reward0;
    this.state1 = state1;
  }

  // A Brain object does all the magic.
  // over time it receives some inputs and some rewards
  // and its job is to set the outputs to maximize the expected reward
  var Brain = function(num_states, num_actions, opt) {
    var opt = opt || {};
    // in number of time steps, of temporal memory
    // the ACTUAL input to the net will be (x,a) temporal_window times, and followed by current x
    // so to have no information from previous time step going into value function, set to 0.
    this.temporal_window = typeof opt.temporal_window !== 'undefined' ? opt.temporal_window : 1; 
    // size of experience replay memory
    this.experience_size = typeof opt.experience_size !== 'undefined' ? opt.experience_size : 30000;
    // number of examples in experience replay memory before we begin learning
    this.start_learn_threshold = typeof opt.start_learn_threshold !== 'undefined'? opt.start_learn_threshold : Math.floor(Math.min(this.experience_size*0.1, 1000)); 
    // gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
    this.gamma = typeof opt.gamma !== 'undefined' ? opt.gamma : 0.8;
    
    // number of steps we will learn for
    this.learning_steps_total = typeof opt.learning_steps_total !== 'undefined' ? opt.learning_steps_total : 100000;
    // how many steps of the above to perform only random actions (in the beginning)?
    this.learning_steps_burnin = typeof opt.learning_steps_burnin !== 'undefined' ? opt.learning_steps_burnin : 3000;
    // what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end
    this.epsilon_min = typeof opt.epsilon_min !== 'undefined' ? opt.epsilon_min : 0.05;
    // what epsilon to use at test time? (i.e. when learning is disabled)
    this.epsilon_test_time = typeof opt.epsilon_test_time !== 'undefined' ? opt.epsilon_test_time : 0.01;
    
    // advanced feature. Sometimes a random action should be biased towards some values
    // for example in flappy bird, we may want to choose to not flap more often
    if(typeof opt.random_action_distribution !== 'undefined') {
      // this better sum to 1 by the way, and be of length this.num_actions
      this.random_action_distribution = opt.random_action_distribution;
      if(this.random_action_distribution.length !== num_actions) {
        console.log('TROUBLE. random_action_distribution should be same length as num_actions.');
      }
      var a = this.random_action_distribution;
      var s = 0.0; for(var k=0;k<a.length;k++) { s+= a[k]; }
      if(Math.abs(s-1.0)>0.0001) { console.log('TROUBLE. random_action_distribution should sum to 1!'); }
    } else {
      this.random_action_distribution = [];
    }
    
    // states that go into neural net to predict optimal action look as
    // x0,a0,x1,a1,x2,a2,...xt
    // this variable controls the size of that temporal window. Actions are
    // encoded as 1-of-k hot vectors
    this.net_inputs = num_states * this.temporal_window + num_actions * this.temporal_window + num_states;
    this.num_states = num_states;
    this.num_actions = num_actions;
    this.window_size = Math.max(this.temporal_window, 2); // must be at least 2, but if we want more context even more
    this.state_window = new Array(this.window_size);
    this.action_window = new Array(this.window_size);
    this.reward_window = new Array(this.window_size);
    this.net_window = new Array(this.window_size);
    
    // create [state -> value of all possible actions] modeling net for the value function
    var layer_defs = [];
    if(typeof opt.layer_defs !== 'undefined') {
      // this is an advanced usage feature, because size of the input to the network, and number of
      // actions must check out. This is not very pretty Object Oriented programming but I can't see
      // a way out of it :(
      layer_defs = opt.layer_defs;
      if(layer_defs.length < 2) { console.log('TROUBLE! must have at least 2 layers'); }
      if(layer_defs[0].type !== 'input') { console.log('TROUBLE! first layer must be input layer!'); }
      if(layer_defs[layer_defs.length-1].type !== 'regression') { console.log('TROUBLE! last layer must be input regression!'); }
      if(layer_defs[0].out_depth * layer_defs[0].out_sx * layer_defs[0].out_sy !== this.net_inputs) {
        console.log('TROUBLE! Number of inputs must be num_states * temporal_window + num_actions * temporal_window + num_states!');
      }
      if(layer_defs[layer_defs.length-1].num_neurons !== this.num_actions) {
        console.log('TROUBLE! Number of regression neurons should be num_actions!');
      }
    } else {
      // create a very simple neural net by default
      layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:this.net_inputs});
      if(typeof opt.hidden_layer_sizes !== 'undefined') {
        // allow user to specify this via the option, for convenience
        var hl = opt.hidden_layer_sizes;
        for(var k=0;k<hl.length;k++) {
          layer_defs.push({type:'fc', num_neurons:hl[k], activation:'relu'}); // relu by default
        }
      }
      layer_defs.push({type:'regression', num_neurons:num_actions}); // value function output
    }
    this.value_net = new convnetjs.Net();
    this.value_net.makeLayers(layer_defs);
    
    // and finally we need a Temporal Difference Learning trainer!
    var tdtrainer_options = {learning_rate:0.01, momentum:0.0, batch_size:64, l2_decay:0.01};
    if(typeof opt.tdtrainer_options !== 'undefined') {
      tdtrainer_options = opt.tdtrainer_options; // allow user to overwrite this
    }
    this.tdtrainer = new convnetjs.SGDTrainer(this.value_net, tdtrainer_options);
    
    // experience replay
    this.experience = [];
    
    // various housekeeping variables
    this.age = 0; // incremented every backward()
    this.forward_passes = 0; // incremented every forward()
    this.epsilon = 1.0; // controls exploration exploitation tradeoff. Should be annealed over time
    this.latest_reward = 0;
    this.last_input_array = [];
    this.average_reward_window = new cnnutil.Window(1000, 10);
    this.average_loss_window = new cnnutil.Window(1000, 10);
    this.learning = true;
  }
  Brain.prototype = {
    random_action: function() {
    },
    policy: function(s) {
    },
    getNetInput: function(xt) {
    },
    forward: function(input_array) {
    },
    backward: function(reward) {
    },
    visSelf: function(elt) {
    }
  }
  
  global.Brain = Brain;
})(deepqlearn);

(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.deepqlearn = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(deepqlearn);