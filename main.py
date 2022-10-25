import matplotlib.pyplot as plt
import numpy as np


def plot_data(data_, alpha_):
    plt.plot(data_[:,0], data_[:,1], label=("Frequency: " + str(30/alpha_) + " hz"))
    plt.legend()
    return

def downsample_data(data_, alpha_):
    downsampled_data_ = data_[::alpha_]
    return downsampled_data_

# def linear_interpolate(orig_data_, int_data_):
#     wp = []
#     alpha_ = round(orig_data_.shape[0]/int_data_.shape[0])
#     for i in range(0, int_data_.shape[0]-1):
#         wp.append([int_data_[i,0], int_data_[i,1]])
#         if (int_data_[i+1,0] - int_data_[i,0] != 0):
#             m = (int_data_[i+1,1] - int_data_[i,1])/(int_data_[i+1,0] - int_data_[i,0])
#             ct = 1
#             while ((ct < alpha_) and (i + ct < int_data_.shape[0])):            
#                 y = int_data_[i,1] + m*(orig_data_[alpha_*(i) + ct,0] - int_data_[i,0])
#                 wp.append([orig_data_[alpha_*(i) + ct,0], y])
#                 ct += 1
#         else:
#             wp.append([int_data_[i,0],int_data_[i,1]])
#             wp.append([int_data_[i+1,0],int_data_[i+1,1]])
#     wp.append([int_data_[-1,0], int_data_[-1,1]])

#     return np.array(wp)   

# def quadratic_interpolation(orig_data_, int_data_):
#     wp = []
#     alpha_ = round(orig_data_.shape[0] / int_data_.shape[0])

#     for i in range(0, int_data_.shape[0]-2,2):
#         ct = 1
#         # flag = 0
#         wp.append([int_data_[i,0], int_data_[i,1]])
#         while(ct < 6):
#             if ct == 3:
#                 wp.append([int_data_[i,0], int_data_[i,1]])
#                 ct = 4
#             if ((int_data_[i,0] - int_data_[i+1,0] != 0) and (int_data_[i,0] - int_data_[i+2,0] != 0) and (int_data_[i+2,0] - int_data_[i+1,0] != 0)):
#                 L0 = ((orig_data_[alpha_*(i) + ct,0] - int_data_[i+1,0])*(orig_data_[alpha_*(i) + ct,0] - int_data_[i+2,0]))/((int_data_[i,0] - int_data_[i+1,0])*(int_data_[i,0] - int_data_[i+2,0]))
#                 L1 = ((orig_data_[alpha_*(i) + ct,0] - int_data_[i,0])*(orig_data_[alpha_*(i) + ct,0] - int_data_[i+2,0]))/((int_data_[i+1,0] - int_data_[i,0])*(int_data_[i+1,0] - int_data_[i+2,0]))
#                 L2 = ((orig_data_[alpha_*(i) + ct,0] - int_data_[i,0])*(orig_data_[alpha_*(i) + ct,0] - int_data_[i+1,0]))/((int_data_[i+2,0] - int_data_[i,0])*(int_data_[i+2,0] - int_data_[i+1,0]))
#                 y = L0*int_data_[i,1] + L1*int_data_[i+1,1] + L2*int_data_[i+2,1]
#                 wp.append([orig_data_[alpha_*(i) + ct,0], y])
            
#             ct += 1
#     print(len(wp))
#     return np.array(wp)

def lin_int(orig_data_, int_data_):
    wp = []
    alpha_ = round(orig_data_.shape[0]/int_data_.shape[0])
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

def quad_int(orig_data_, int_data_):
    wp  =[]
    alpha_ = round(orig_data_.shape[0] / int_data_.shape[0])

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

def cub_int(orig_data_, int_data_):
    wp  =[]
    alpha_ = round(orig_data_.shape[0] / int_data_.shape[0])    

    n = int_data_.shape[0]
    p = n - 1
    Y = np.zeros((4*p,1))
    M = np.zeros((4*p, 4*p))

    # Fixing the 0th and (2p-1)th elements
    # Y[0,:] = int_data_[0,1]
    # Y[2*p - 1,:] = int_data_[-1,1]
    
    M[0,0] = 0
    M[0,1] = 0
    M[0,2] = 0
    M[0,3] = 1
    M[2*p - 1,4*p - 4] = (alpha_*p)**3
    M[2*p - 1,4*p - 3] = (alpha_*p)**2
    M[2*p - 1,4*p - 2] = (alpha_*p)
    M[2*p - 1,4*p - 1] = 1

    for i in range(1,p):

        # Feeding the M matrix of shape (4*n, 4*n). Unfilled cells will be 0s.
        t = alpha_*i
        M[2*i-1,4*i-4] = (alpha_*i)**3
        M[2*i,4*i] = (alpha_*i)**3
        
        M[2*i-1,4*i-3] = (alpha_*i)**2
        M[2*i,4*i+1] = (alpha_*i)**2
        
        M[2*i-1,4*i-2] = (alpha_*i)
        M[2*i,4*i+2] = (alpha_*i)
        
        M[2*i-1,4*i-1] = 1
        M[2*i,4*i+3] = 1   

        # First differential coefficients
        M[2*p + i - 1, 4*i - 4] = 3*((alpha_*i)**2)
        M[2*p + i - 1, 4*i - 3] = 2*(alpha_*i)
        M[2*p + i - 1, 4*i - 2] = 1
        M[2*p + i - 1, 4*i] = -3*((alpha_*i)**2)
        M[2*p + i - 1, 4*i + 1] = -2*(alpha_*i)
        M[2*p + i - 1, 4*i + 2] = -1

        # Second differential coefficients
        M[3*p + i - 2, 4*i - 4] = 6*(alpha_*i)
        M[3*p + i - 2, 4*i - 3] = 2
        M[3*p + i - 2, 4*i] = -6*(alpha_*i)
        M[3*p + i - 2, 4*i + 1] = -2       

    # Boundary conditions
    M[4*p-2, 0] = 0
    M[4*p-2, 1] = 2
    M[4*p-1, 4*p-4] = 6*(alpha_*p)
    M[4*p-1, 4*p-1] = 2

    print(np.linalg.det(M))

    print(np.linalg.det(np.linalg.inv(M)))

    return

def cubic_interpolation(orig_data_, int_data_):
    n = int_data_.shape[0]
    p = n - 1
    Y = np.zeros((4*p,1))
    M = np.zeros((4*p, 4*p))

    # Fixing the 0th and (2p-1)th elements
    Y[0,:] = int_data_[0,1]
    Y[2*p - 1,:] = int_data_[-1,1]
    
    M[0,0] = int_data_[0,0]**3
    M[0,1] = int_data_[0,0]**2
    M[0,2] = int_data_[0,0]
    M[0,3] = 1
    M[2*p - 1,4*p - 4] = int_data_[-1,0]**3
    M[2*p - 1,4*p - 3] = int_data_[-1,0]**2
    M[2*p - 1,4*p - 2] = int_data_[-1,0]
    M[2*p - 1,4*p - 1] = 1

    for i in range(1, p):

        # Feeding the Y vector of shape (4*n,1). Unfilled cells will be 0s.
        Y[2*i - 1 : 2*i + 1,:] = int_data_[i,1]
        # Feeding the M matrix of shape (4*n, 4*n). Unfilled cells will be 0s.
        M[2*i-1,4*i-4] = int_data_[i,0]**3
        M[2*i,4*i] = int_data_[i,0]**3
        
        M[2*i-1,4*i-3] = int_data_[i,0]**2
        M[2*i,4*i+1] = int_data_[i,0]**2
        
        M[2*i-1,4*i-2] = int_data_[i,0]
        M[2*i,4*i+2] = int_data_[i,0]
        
        M[2*i-1,4*i-1] = 1
        M[2*i,4*i+3] = 1

        # First differential coefficients
        M[2*p + i - 1, 4*i - 4] = 3*(int_data_[i,0]**2)
        M[2*p + i - 1, 4*i - 3] = 2*int_data_[i,0]
        M[2*p + i - 1, 4*i - 2] = 1
        M[2*p + i - 1, 4*i] = -3*(int_data_[i,0]**2)
        M[2*p + i - 1, 4*i + 1] = -2*int_data_[i,0]
        M[2*p + i - 1, 4*i + 2] = -1

        # Second differential coefficients
        M[3*p + i - 2, 4*i - 4] = 6*int_data_[i,0]
        M[3*p + i - 2, 4*i - 3] = 2
        M[3*p + i - 2, 4*i] = -6*int_data_[i,0]
        M[3*p + i - 2, 4*i + 1] = -2

    # Boundary conditions
    M[4*p-2, 0] = 6*int_data_[0,0]
    M[4*p-2, 1] = 2
    M[4*p-1, 4*p-4] = 6*int_data_[-1,0]
    M[4*p-1, 4*p-1] = 2

    # print(np.linalg.det(M))

    return #np.linalg.pinv(M) @ Y

def ci(orig_data_, int_data_):

    alpha_ = round(orig_data_.shape[0]/int_data_.shape[0])
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

    # Get Downsampled Data
    data_10hz = downsample_data(waypoints, 3)
    data_01hz = downsample_data(waypoints, 30)
    data_p2hz = downsample_data(waypoints, 150)

    ## LINEAR INTERPOLATION
    lin_10hz = lin_int(waypoints, data_10hz)
    lin_01hz = lin_int(waypoints, data_01hz)
    lin_p2hz = lin_int(waypoints, data_p2hz)
    print(get_error(waypoints, lin_10hz))
    print(get_error(waypoints, lin_01hz))
    print(get_error(waypoints, lin_p2hz))
    plot_data(waypoints, 1)
    plot_data(lin_10hz, 3)
    plot_data(lin_01hz, 30)
    plot_data(lin_p2hz, 150)

    ## QUADRATIC INTERPOLATION
    quad_10hz = quad_int(waypoints, data_10hz)
    quad_01hz = quad_int(waypoints, data_01hz)
    quad_p2hz = quad_int(waypoints, data_p2hz)
    print(get_error(waypoints, quad_10hz))
    print(get_error(waypoints, quad_01hz))
    print(get_error(waypoints, quad_p2hz))
    plot_data(waypoints, 1)
    plot_data(quad_10hz, 3)
    plot_data(quad_01hz, 30)
    plot_data(quad_p2hz, 150)

    ## CUBIC INTERPOLATION
    cub_10hz = ci(waypoints, data_10hz)
    cub_01hz = ci(waypoints, data_01hz)
    cub_p2hz = ci(waypoints, data_p2hz)
    print(get_error(waypoints, cub_10hz))
    print(get_error(waypoints, cub_01hz))
    print(get_error(waypoints, cub_p2hz))
    plot_data(waypoints, 1)
    plot_data(cub_10hz, 3)
    plot_data(cub_01hz, 30)
    plot_data(cub_p2hz, 150)

    plt.show(block=True)