import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

def angle(x1,x0,y1,y0): # Function that finds the angle between two points, (x0,y0) and (x1,y1)
  dx = x1-x0
  dy = y1-y0
  angle = np.arctan(abs(dy)/abs(dx)) 
  if dx<0:
    if dy<0: 
      return(angle-np.pi)
    else: 
      return(np.pi-angle)
  else:
    if dy<0:
      return(-angle) 
    else:
      return(angle)
      
      
def function(z, t, y):    # ODE functions
  phi, r, s = z
  M, L = y

  dpdt = L/r**2
  drdt = s
  dsdt = -(L**2)*(3.0*M-r)/(r**4) 
  return [dpdt, drdt, dsdt]\
  
 
celestial = im.open(’celestial.jpeg’) 
f = 120                             # Focal length, Range of values of L, Time steps, Height and width of image
l = np.linspace(-1.414,-30,100) 
t = np.linspace(0,5000,100)
x, y = celestial.size
alpha, alphadash = [], []

for i in l:
  yo = [1.5, i]                                             # Black hole constants [M, L]
  zo = [np.pi, 20, -0.13].                                  # Camera location [phi0, r0, s0]
  z = scipy.integrate.odeint(function,zo,t, args=(yo,))     # Solve ODE
  phi, r, s = z[:,0], z[:,1], z[:,2]
  x1, y1 = r * np.cos(phi), r * np.sin(phi) 
  alpha.append(angle(x1[1],x1[0],y1[1],y1[0])) 
  alphadash.append(angle(x1[-1],x1[0],y1[-1],y1[0]))
  
  
a1, ap1 = [],[]

for k in range(len(alpha)-1): # Removes negative alpha values 
  if alpha[k] > 0:
    a1.append(alpha[k]) 
    ap1.append(alphadash[k])
    
blackhole = im.new("RGB", (x,y)) 
for i in range(x):
  for j in range(y):
    dx,dy = x/2 - i , y/2 - j                             # Lengths on camera image, (dx,dy,dxy)
    dxy = np.sqrt((dx)**2+(dy)**2)
    a = np.arctan(dxy/f)                                  # Finds angle a = Alpha
    if dxy == 0:                                          # Finds angle b = Beta
      b = 0
    elif i > x/2:
       b = np.arccos(dy/dxy) 
    else:
       b = -np.arccos(dy/dxy)
    if a >= a1[-1]:                                       # Finds angle ap = Alphaprime
      ap = a 
    else:
      ap = np.interp( a, a1, ap1, left=0, right=None, period= 2*np.pi)     # Finds angle bp = Betaprime
    bp = -b

    dxyp = f * np.tan(ap) # Lengths on sky image, (dxp,dyp,dxyp)
    dxp, dyp = dxyp * np.sin(bp), dxyp * np.cos(bp)
    ip = np.abs( np.round(x/2 + dxp)) # Matching up the corresponding pixels 
    jp = np.abs(np.round(y/2 + dyp))
    ip.astype(int)
    jp.astype(int)
    if a1[0] > a : # No photons travel from blackhole
      blackhole.putpixel((i,j), (0,0,0)) 
    elif ip < x and jp < y:
      match = celestial.getpixel((ip,jp))
      blackhole.putpixel((i,j), match) 
blackhole.show()
plt.show() 
plt.savefig(’blackhole.png’)
