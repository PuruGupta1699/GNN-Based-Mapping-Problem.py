np.random.seed(16)

def Adjacency(state):
    adj = []
    dis = []
    for j in range(20):
        dis.append([state[j][-2],state[j][-1],j])
    for j in range(20):
        f = []
        for r in range(len(dis)):
            f.append([(dis[r][0]-dis[j][0])**2+(dis[r][1]-dis[j][1])**2,r])
        f.sort(key=lambda x:x[0])
        y = []
        for r in range(4):
            y.append(f[r][1])
        y = to_categorical(y,num_classes=20)
        adj.append(y)
    return adj

def observation(state1,state2):
    state = []
    for j in range(20):
        state.append(np.hstack(((state1[j][0:11,0:11,1]-state1[j][0:11,0:11,5]).flatten(),state2[j][-1:-3:-1])))
    return state
