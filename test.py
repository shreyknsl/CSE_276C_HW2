import  numpy as np

waypoints = np.loadtxt("./waypoints.csv", delimiter=",")
wp = []
i = 0
for p in waypoints:
    wp.append(p)
    print(p.shape)
    print(p.reshape((2,1)).shape)
    print(p[0])
    print(p[1])
    if i == 2: break
    i += 1
print(wp)
print(np.array(wp))