import matplotlib.pyplot as plt
import numpy as np


def plot_data(data_, alpha_):
    plt.plot(data_[:,0], data_[:,1], label=("Interpolated from " + str(30/alpha_) + " hz to 30Hz"))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show(block=True)
    return

def downsample_data(data_, alpha_):
    downsampled_data_ = data_[::alpha_]
    return downsampled_data_


def lin_int(int_data_, alpha_):
    wp = []

    for i in range(0, int_data_.shape[0]-1):
        f0 = int_data_[i]
        wp.append(f0)
        f1 = int_data_[i+1]
        m = (f1 - f0)/(alpha_)
        ct = 1
        while (ct < alpha_):
            p = f0 + m*(ct)
            wp.append([p[0], p[1]])
            ct += 1
    wp.append(int_data_[-1])
    return np.array(wp)

def quad_int(int_data_, alpha_):
    wp  =[]

    for i in range(0, int_data_.shape[0]-2, 2):
        t0 = i*alpha_
        t1 = t0 + alpha_
        t2 = t0 + 2*alpha_
        f0 = int_data_[i]
        f1 = int_data_[i+1]
        f2 = int_data_[i+2]
        wp.append(f0)
        ct = 1
        while (ct < 2*alpha_):
            
            if ct == alpha_:
                wp.append(int_data_[i+1])
                ct = alpha_ + 1

            t = t0 + ct
            L0 = ((t - t1)*(t - t2))/((t0 - t1)*(t0 - t2))
            L1 = ((t - t0)*(t - t2))/((t1 - t0)*(t1 - t2))
            L2 = ((t - t0)*(t - t1))/((t2 - t0)*(t2 - t1))
            p = L0*f0 + L1*f1 + L2*f2
            wp.append([p[0], p[1]])

            ct += 1

    return  np.array(wp)

def cub_int(int_data_, alpha_):

    n = int_data_.shape[0] - 1
    A = 2*np.eye(n+1)
    dx = np.zeros((n+1,1))     #d[0] = d[n] = 0
    dy = np.zeros((n+1,1))
    A[0,1] = 0
    A[-1,-2] = 0

    for i in range(1,n):

        t0 = alpha_*(i-1)
        t1 = alpha_*(i)
        t2 = alpha_*(i+1)

        fx0 = int_data_[i-1,0]
        fx1 = int_data_[i,0]
        fx2 = int_data_[i+1,0]

        fy0 = int_data_[i-1,1]
        fy1 = int_data_[i,1]
        fy2 = int_data_[i+1,1]        

        h1 = t1 - t0
        h2 = t2 - t1

        mu_ = h1/(h1 + h2)
        lambda_ = 1 - mu_

        A[i, i-1] = mu_
        A[i, i+1] = lambda_

        Lx0 = fx0/((t0-t1)*(t0-t2))
        Lx1 = fx1/((t1-t0)*(t1-t2))
        Lx2 = fx2/((t2-t0)*(t2-t1))

        Ly0 = fy0/((t0-t1)*(t0-t2))
        Ly1 = fy1/((t1-t0)*(t1-t2))
        Ly2 = fy2/((t2-t0)*(t2-t1))        
        
        dx[i] = 6*(Lx0 + Lx1 + Lx2)
        dy[i] = 6*(Ly0 + Ly1 + Ly2)

    Mx = np.linalg.inv(A) @ dx
    
    My = np.linalg.inv(A) @ dy
    np.savetxt('My.csv', My)
    wp = []
    
    for i in range(1,n):
        x0 = int_data_[i-1,0]
        x1 = int_data_[i,0]
        y0 = int_data_[i-1,1]
        y1 = int_data_[i,1]        
        wp.append([x0,y0])
        ct = 1
        
        t0 = (i-1)*alpha_
        t1 = t0 + alpha_

        h1 = t1 - t0

        while(ct < alpha_):
            t = t0 + ct
            px = Mx[i-1]*((t1 - t)**3)/(6*(h1)) + Mx[i]*((t - t0)**3)/(6*(h1)) + (x0 - (Mx[i-1]*(h1**2))/6)*((t1 - t)/h1) + (x1 - (Mx[i]*(h1**2))/6)*((t - t0)/h1)
            py = My[i-1]*((t1 - t)**3)/(6*(h1)) + My[i]*((t - t0)**3)/(6*(h1)) + (y0 - (My[i-1]*(h1**2))/6)*((t1 - t)/h1) + (y1 - (My[i]*(h1**2))/6)*((t - t0)/h1)
            wp.append([float(px),float(py)])
            ct += 1

    wp.append([int_data_[i-1,0], int_data_[i-1,1]])

    return np.array(wp)

def get_error(orig_data_, int_data_):
    orig_data_ = orig_data_[:int_data_.shape[0]]
    err = np.sum(np.sqrt(np.square(orig_data_[:,0] - int_data_[:,0]) + np.square(orig_data_[:,1] - int_data_[:,1])))
    return err

if __name__ == "__main__":
    
    waypoints = np.loadtxt("./waypoints.csv", delimiter=",")

    ## Get Downsampled Data
    alpha_10 = 3
    alpha_01 = 30
    alpha_p2 = 150
    data_10hz = downsample_data(waypoints, alpha_10)
    data_01hz = downsample_data(waypoints, alpha_01)
    data_p2hz = downsample_data(waypoints, alpha_p2)

    ## GROUND TRUTH
    plot_data(waypoints,1)

    ## LINEAR INTERPOLATION
    lin_10hz = lin_int(data_10hz, alpha_10)
    print(get_error(waypoints, lin_10hz))
    plot_data(lin_10hz, 3)
    lin_01hz = lin_int(data_01hz, alpha_01)
    print(get_error(waypoints, lin_01hz))
    plot_data(lin_01hz, 30)
    lin_p2hz = lin_int(data_p2hz, alpha_p2)
    print(get_error(waypoints, lin_p2hz))
    plot_data(lin_p2hz, 150)

    ## QUADRATIC INTERPOLATION
    quad_10hz = quad_int(data_10hz, alpha_10)
    print(get_error(waypoints, quad_10hz))
    plot_data(quad_10hz, 3)
    quad_01hz = quad_int(data_01hz, alpha_01)
    print(get_error(waypoints, quad_01hz))
    plot_data(quad_01hz, 30)
    quad_p2hz = quad_int(data_p2hz, alpha_p2)
    print(get_error(waypoints, quad_p2hz))
    plot_data(quad_p2hz, 150)

    ## CUBIC INTERPOLATION
    cub_10hz = cub_int(data_10hz, alpha_10)
    print(get_error(waypoints, cub_10hz))
    plot_data(cub_10hz, 3)
    cub_01hz = cub_int(data_01hz, alpha_01)
    print(get_error(waypoints, cub_01hz))
    plot_data(cub_01hz, 30)
    cub_p2hz = cub_int(data_p2hz, alpha_p2)
    print(get_error(waypoints, cub_p2hz))
    plot_data(cub_p2hz, 150)