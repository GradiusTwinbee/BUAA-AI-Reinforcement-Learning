from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998
import tensorflow as tf
from future import standard_library

standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
import itertools
from malmo import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import malmoutils
import pickle as pk

import plotting
from collections import deque, namedtuple
import numpy as np
from random import randint

# if sys.version_info[0] == 2:
#     # Workaround for https://github.com/PythonCharmers/python-future/issues/262
#     import Tkinter as tk
# else:
#     import tkinter as tk

malmoutils.fix_print()

actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
actionValue = [9,-9,1,-1]
my_pos = -1
my_path = [False] * 81
my_count = 0
COUNT_MAX = 20
my_life = 2
my_success = 0
# ===========================================================================================================================


class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """

    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[9, 9, 1], dtype=tf.uint8)
            # self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = self.input_state
            # self.output = tf.image.crop_to_bounding_box(self.output, 0, 0, 10, 10)
            # self.output = tf.image.resize_images(
            #     self.output, [10, 10], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [10, 10, 1] Maze RGB State
        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        val = sess.run(self.output, {self.input_state: state})
        # print("!!!!!!!!!!!!!!!!!!!",val)
        return val

def gridProcess(state):
    global my_path,my_pos,my_count
    msg = state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor10x10', 0)
    if my_pos != grid.index("emerald_block"):
        my_pos = grid.index("emerald_block")
        my_count = 0
    else:
        my_count += 1
    print(my_pos)
    my_path[my_pos] = True
    obs = np.array(grid)
    obs = np.reshape(obs, [9, 9, 1])
    
    obs[obs == "air"] = 0
    obs[obs == "beacon"] = 0
    obs[obs == "carpet"] = 1
    obs[obs == "fire"] = 2
    obs[obs == "netherrack"] = 2
    obs[obs == "sea_lantern"] = 3
    obs[obs == "emerald_block"] = 4
    obs[obs == "grass"] = 5
    obs[obs == "glass"] = 6
    obs[obs == "human"] = 7
    # print("Here is obs", obs)
    # exit(0)
    return obs


class Estimator():
    """Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 1 RGB frames of shape 10, 10 each
        self.X_pl = tf.placeholder(shape=[None, 9, 9, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 10.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(actionSet))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 1, 10, 10, 1]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        # print("S's shape:",s.shape)
        return sess.run(self.predictions, {self.X_pl: s})



def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)

        return A

    return policy_fn


def deep_q_learning(sess,
                    agent_host,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=50000,
                    replay_memory_init_size=5000,
                    update_target_estimator_every=1000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=8000,
                    batch_size=32):
    global my_path
    global my_pos
    global my_count
    global my_life
    mission_file = agent_host.getStringArgument('mission_file')
    mission_file = os.path.join(mission_file, "Maze0.xml")
    currentMission = mission_file
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.setViewpoint(2)
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available
    # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10001))

    max_retries = 3
    agentID = 0
    expID = 'Deep_q_learning memory'

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "check")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver(max_to_keep=1)
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(actionSet))

    my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                "save_%s-rep" % (expID))

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s" % (expID))
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2.5)

    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        print("Sleeping")
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
    print()
    agent_host.sendCommand("look -1")
    agent_host.sendCommand("look -1")
    print("Populating replay memory...")

    while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
        print("Sleeping....")
        world_state = agent_host.peekWorldState()
    # Populate the replay memory with initial experience

    while world_state.number_of_observations_since_last_state <= 0 and world_state.is_mission_running:
        # print("Sleeping")
        time.sleep(0.1)
        world_state = agent_host.peekWorldState()
    
    state = gridProcess(world_state)  # MALMO ENVIRONMENT Grid world NEEDED HERE/ was env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    currentMission = mission_file
    i_episode = 0
    i = 100
    while True:
        if my_life == 0:
            i_episode += 1
            my_life = 2
        if i_episode > 159:
            break
        print("%s-th episode" % i_episode)
        my_path = [False]*81
        my_pos = -1
        my_count = 0
        if True:
            mission_file = agent_host.getStringArgument('mission_file')
            mazeNum = i_episode
            mission_file = os.path.join(mission_file,"Maze%s.xml"%mazeNum)
            currentMission = mission_file

            with open(mission_file, 'r') as f:
                print("Loading mission from %s" % mission_file)
                mission_xml = f.read()
                my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.removeAllCommandHandlers()
            my_mission.allowAllDiscreteMovementCommands()
            # my_mission.requestVideo(320, 240)
            my_mission.forceWorldReset()
            my_mission.setViewpoint(2)
            my_clients = MalmoPython.ClientPool()
            my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

            max_retries = 3
            agentID = 0
            expID = 'Deep_q_learning '

            my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                        "save_%s-rep%d" % (expID, i))

            for retry in range(max_retries):
                try:
                    agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i))
                    break
                except RuntimeError as e:
                    if retry == max_retries - 1:
                        print("Error starting mission:", e)
                        exit(1)
                    else:
                        time.sleep(2.5)

            world_state = agent_host.getWorldState()
            print("Waiting for the mission to start", end=' ')
            while not world_state.has_mission_begun:
                print(".", end="")
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)
        agent_host.sendCommand("look -1")
        agent_host.sendCommand("look -1")
        
        while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        # world_state = agent_host.getWorldState()
        state = gridProcess(world_state)  # MalmoGetWorldState?
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}".format(
                t, total_t, i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # next_state, reward, done, _ = env.step(actionSet[action]) # Malmo AgentHost send command?
            # print("Sending command: ", actionSet[action])
            if my_pos % 9 >= 5:
                if action == 0:
                    action = 1
                elif action == 1:
                    action = 0
                elif action == 2:
                    action = 3
                elif action == 3:
                    action == 2
            try:
                if not my_path[my_pos + actionValue[action]]:
                    agent_host.sendCommand(actionSet[action])
                    time.sleep(0.1)
            except IndexError:
                pass
            world_state = agent_host.peekWorldState()
            num_frames_seen = world_state.number_of_video_frames_since_last_state

            while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
                world_state = agent_host.peekWorldState()

            done = not world_state.is_mission_running
            if my_count>COUNT_MAX:
                my_path = [False] * 81
            print(" IS MISSION FINISHED? ", done)
            if world_state.is_mission_running:
                while world_state.number_of_rewards_since_last_state <= 0:
                    time.sleep(0.1)
                    world_state = agent_host.peekWorldState()
                reward = world_state.rewards[-1].getValue()
                print("Just received the reward: %s on action: %s " % (reward, actionSet[action]))

                while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                    world_state = agent_host.peekWorldState()
                # world_state = agent_host.getWorldState()

                if world_state.is_mission_running:
                    next_state = gridProcess(world_state)
                    next_state = state_processor.process(sess, next_state)
                    next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                else:
                    print("Mission finished prematurely")
                    next_state = state
                    done = not world_state.is_mission_running


                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if done:
                    print("End of episode")
                    break
                state = next_state
                total_t += 1

            if done:
                if len(world_state.rewards)>0:
                    reward = world_state.rewards[-1].getValue()
                else:
                    print("IDK no reward")
                    reward= 0
                print("Just received the reward: %s on action: %s " % (reward, actionSet[action]))

                next_state = state

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                print("End of Episode")
                break


        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])
    return stats


# Main body=======================================================

agent_host = MalmoPython.AgentHost()

# Find the default mission file by looking next to the schemas folder:
schema_dir = None
try:
    # schema_dir = os.environ['mazes']
    schema_dir = "mazes"
except KeyError:
    print("MALMO_XSD_PATH not set? Check environment.")
    exit(1)
mission_file = os.path.abspath(schema_dir)  # Integration test path
# if not os.path.exists(mission_file):
#     mission_file = os.path.abspath(os.path.join(schema_dir, '..',
#                                                 'sample_missions', 'mazes1'))  # Install path
if not os.path.exists(mission_file):
    print("Could not find Maze.xml under MALMO_XSD_PATH")
    exit(1)

# add some args
agent_host.addOptionalStringArgument('mission_file',
                                     'Path/to/file from which to load the mission.', mission_file)
# agent_host.addOptionalFloatArgument('alpha',
#                                     'Learning rate of the Q-learning agent.', 0.1)
# agent_host.addOptionalFloatArgument('epsilon',
#                                     'Exploration rate of the Q-learning agent.', 0.01)
# agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 0.99)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.LATEST_REWARD_ONLY)
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
malmoutils.parse_command_line(agent_host)

tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format("DeepQLearning"))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    agent_host,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=160,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50,
                                    update_target_estimator_every=100,
                                    epsilon_start=0.2,
                                    epsilon_end=0.2,
                                    epsilon_decay_steps=50000,
                                    discount_factor=0.99,
                                    batch_size=32):
        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
        if stats.episode_rewards[-1] > 0:
            my_life = 0
            my_success += 1
        else:
            my_life -= 1
        

print("success:")
print(my_success)
pk.dump(my_success,open("success.pkl","wb"))
# ======================================================================================
