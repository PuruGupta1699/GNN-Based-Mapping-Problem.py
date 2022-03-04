import sys

import wandb
from omegaconf import OmegaConf as omg
import torch

from agent import Agent
from trainer import Trainer, reward_fn
from data import DataGenerator


def load_conf():
    """Quick method to load configuration (using OmegaConf). By default,
    configuration is loaded from the default config file (config.yaml).
    Another config file can be specific through command line.
    Also, configuration can be over-written by command line.

    Returns:
        OmegaConf.DictConfig: OmegaConf object representing the configuration.
    """
    default_conf = omg.create({"config" : "config.yaml"})

    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    yaml_file = omg.merge(default_conf, cli_conf).config

    yaml_conf = omg.load(yaml_file)

    return omg.merge(default_conf, yaml_conf, cli_conf)


def main():
    conf = load_conf()
    wandb.init(project=conf.proj_name, config=dict(conf))

    agent = Agent(embed_hidden=conf.embed_hidden, enc_stacks=conf.enc_stacks, ff_hidden=conf.ff_hidden, enc_heads=conf.enc_heads, query_hidden=conf.query_hidden, att_hidden=conf.att_hidden, crit_hidden=conf.crit_hidden, n_history=conf.n_history, p_dropout=conf.p_dropout)
    wandb.watch(agent)

    dataset = DataGenerator()

    trainer = Trainer(conf, agent, dataset)
    trainer.run()

    # Save trained agent
    torch.save(agent.state_dict(), conf.model_path)

    path="data/battle_model"
map_size=100
capacity = 200000
batch_size = 256
totalTime = 0
TAU = 0.01     
LRA = 0.0001        
param = None
alpha = 0.6
GAMMA = 0.96
n_episode = 100000
max_steps = 300
episode_before_train = 200
n_agent=20
magent.utility.init_logger("battle")
env = magent.GridWorld("battle", map_size=30)
env.set_render_dir("build/render")
handles = env.get_handles()
sess = tf.Session()
K.set_session(sess)
n = len(handles)
n_actions=env.get_action_space(handles[0])[0]
i_episode=0
buff=ReplayBuffer(capacity)
l=40

print(env.get_action_space(handles[0])[0])
print(env.get_action_space(handles[1])[0])
#f = open('log.txt','w')

######build the model#########
cnn = MLP()
m1 = MultiHeadsAttModel(l=4)
m2 = MultiHeadsAttModel(l=4)
q_net = Q_Net(action_dim = 9)
vec = np.zeros((1,4))
vec[0][0] = 1

In= []
for j in range(n_agent):
    In.append(Input(shape=[123]))
    In.append(Input(shape=(4,20)))
In.append(Input(shape=(1,4)))
feature = []
for j in range(n_agent):
    feature.append(cnn(In[j*2]))

feature_ = merge(feature,mode='concat',concat_axis=1)

relation1 = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],feature_])
    relation1.append(m1([T,T,T,In[40]]))

relation1_ = merge(relation1,mode='concat',concat_axis=1)

relation2 = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],relation1_])
    relation2.append(m2([T,T,T,In[40]]))

V = []
for j in range(n_agent):
    V.append(q_net([feature[j],relation1[j],relation2[j]]))

model = Model(input=In,output=V)
model.compile(optimizer=Adam(lr = 0.00003), loss='mse')
model.summary()

######build the target model#########
cnn_t = MLP()
m1_t = MultiHeadsAttModel(l=4)
m2_t = MultiHeadsAttModel(l=4)
q_net_t = Q_Net(action_dim = 9)
In_t= []
for j in range(n_agent):
    In_t.append(Input(shape=[123]))
    In_t.append(Input(shape=(4,20)))
In_t.append(Input(shape=(1,4)))

feature_t = []
for j in range(n_agent):
    feature_t.append(cnn_t(In_t[j*2]))

feature_t_ = merge(feature_t,mode='concat',concat_axis=1)

relation1_t = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],feature_t_])
    relation1_t.append(m1_t([T,T,T,In_t[40]]))

relation1_t_ = merge(relation1_t,mode='concat',concat_axis=1)

relation2_t = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],relation1_t_])
    relation2_t.append(m2_t([T,T,T,In_t[40]]))

V_t = []
for j in range(n_agent):
    V_t.append(q_net_t([feature_t[j],relation1_t[j],relation2_t[j]]))

model_t = Model(input=In_t,output=V_t)

tf_model = DeepQNetwork(env, handles[1], 'trusty-battle-game-l', use_conv=True)
tf_model.load("data/battle_model", 0, 'trusty-battle-game-l')
###########playing#############
while i_episode<n_episode:
    alpha*=0.996
    if alpha<0.01:
        alpha=0.01
    print(i_episode)
    i_episode=i_episode+1
    env.reset()
    #env.add_walls(method="random", n=map_size * map_size * 0.03)
    env.add_agents(handles[0], method="random", n=20)
    env.add_agents(handles[1], method="random", n=12)
    step_ct = 0
    done = False
    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    action = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    steps = 0
    score = 0
    loss = 0
    dead = [0,0]
    
    while steps<max_steps:
        steps+=1
        i=0
        obs[i] = env.get_observation(handles[i])
        adj = Adjacency(obs[i][1])
        flat_ob = observation(obs[i][0],obs[i][1])
        ob=[]
        for j in range(n_agent):
            ob.append(np.asarray([flat_ob[j]]))
            ob.append(np.asarray([adj[j]]))
        ob.append(np.asarray([vec]))
        acts = model.predict(ob)
        action[i]=np.zeros(n_agent,dtype = np.int32)
        for j in range(n_agent):
            if np.random.rand()<alpha:
                action[i][j]=random.randrange(n_actions)
            else:
                action[i][j]=np.argmax(acts[j])
        env.set_action(handles[i], action[i])

        obs[1] = env.get_observation(handles[1])
        ids[1] = env.get_agent_id(handles[1])
        acts = tf_model.infer_action(obs[1], ids[1], 'e_greedy')
        env.set_action(handles[1], acts)
        done = env.step()
        
        next_obs = env.get_observation(handles[0])
        flat_next_obs = observation(next_obs[0],next_obs[1])
        rewards = env.get_reward(handles[0])
        score += sum(rewards)
        if steps%3 ==0:
            buff.add(flat_ob, action[0], flat_next_obs, rewards, done, adj)

        if (i_episode-1) % 10 ==0:
            env.render()
        if max_steps == steps:
            print(dead[0],end='\t')
            print(dead[1],end='\t')
            print(score/300,end='\t')
            #f.write(str(dead[i])+'\t'+str(score[i]/300)+'\t')
            #f.write(str(loss/100)+'\n')
            print(loss/100,end='\n')
        env.clear_dead()

        ############add to n_agent##############
        idd = n_agent - len(env.get_agent_id(handles[0]))
        if idd>0:
            env.add_agents(handles[0], method="random", n=idd)
            dead[0]+=idd
        idd = 12 - len(env.get_agent_id(handles[1]))
        if idd>0:
            env.add_agents(handles[1], method="random", n=idd)
            dead[1]+=idd

        if i_episode < episode_before_train:
            continue
        if steps%3 != 0:
            continue
        #############training###########
        batch = buff.getBatch(128)
        states,actions,rewards,new_states,dones,adj=[],[],[],[],[],[]
        for i_ in  range(n_agent*2+1):
            states.append([])
            new_states.append([])
        for e in batch:
            for j in range(n_agent):
                states[j*2].append(e[0][j])
                states[j*2+1].append(e[5][j])
                new_states[j*2].append(e[2][j])
                new_states[j*2+1].append(e[5][j])
            states[40].append(vec)
            new_states[40].append(vec)
            actions.append(e[1])
            rewards.append(e[3])
            dones.append(e[4])
        
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        
        for i_ in  range(n_agent*2+1):
            states[i_]=np.asarray(states[i_])
            new_states[i_]=np.asarray(new_states[i_])

        q_values = model.predict(states)
        target_q_values = model_t.predict(new_states)

        for k in range(len(batch)):
            if dones[k]:
                for j in range(n_agent):
                    q_values[j][k][actions[k][j]] = rewards[k][j]
            else:
                for j in range(n_agent):
                    q_values[j][k][actions[k][j]] =rewards[k][j] + GAMMA*np.max(target_q_values[j][k])

        history=model.fit(states, q_values,epochs=1,batch_size=128,verbose=0)
        his=0
        for (k,v) in history.history.items():
            his+=v[0]
        loss+=(his/20)
        #########train target model#########
        weights = cnn.get_weights()
        target_weights = cnn_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        cnn_t.set_weights(target_weights)

        weights = q_net.get_weights()
        target_weights = q_net_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        q_net_t.set_weights(target_weights)

        weights = m1.get_weights()
        target_weights = m1_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        m1_t.set_weights(target_weights)

        weights = m2.get_weights()
        target_weights = m2_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        m2_t.set_weights(target_weights)

        #######save model###############
    model.save('gdn.h5')

if __name__ == "__main__":
    main()
