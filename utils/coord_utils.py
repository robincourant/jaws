

# A Class converts different coordinate defintions

''' 

without any indication, the class takes z-forward (** SLAM ** method) 
as default coordinate system while all converting process

supported types: 
OpenGL, Unity3D, FreeD, Unreal, SLAM
By Xi WANG 29/10/2019

- supported as matrix (31/05/2022)
'''

import numpy as np
from scipy.spatial.transform import Rotation
import math
from enum import Enum

class CoordType(Enum):
  SLAM    = 0 #openCV
  OpenGL  = 1
  Unity3D = 2
  FreeD   = 3
  UE      = 4
  INV_SLAM= 5

# precomputed transform matrices
s_M_opgl = np.transpose(np.matrix([[1,0,0],[0,-1,0],[0,0,-1]]))
s_M_u3d  = np.transpose(np.matrix([[1,0,0],[0,-1,0],[0,0,1]]))
s_M_fd   = np.transpose(np.matrix([[1,0,0],[0,0,1],[0,-1,0]]))
s_M_ue   = np.transpose(np.matrix([[0,0,1],[1,0,0],[0,-1,0]]))

# test 
s_M_invs   = np.transpose(np.matrix([[-1,0,0],[0,-1,0],[0,0,-1]]))

class coordCvtor():

  # R is Rotation in SciPy
  # t is the Matrix/Array in Numpy
  def __init__(self, R, t, cType=CoordType.SLAM):

    if(cType==CoordType.SLAM):
      self.R = R
      self.t = t
      return

    R_cvt, t_cvt = self.convert(R,t,cTypeIn=cType)
    self.R = R_cvt
    self.t = t_cvt

  '''
    Ctors
  '''

  @classmethod
  def from_SLAM_mat(cls, M):
    R = Rotation.from_matrix(M[:3,:3])
    t = np.matrix([M[0,3],M[1,3],M[2,3]])
    return cls(R, t)

  @classmethod
  def from_SLAM(cls, R, t):
    return cls(R,t)

  @classmethod
  def from_OpenGL_mat(cls, M):
    R = Rotation.from_matrix(M[:3,:3])
    t = np.matrix([M[0,3],M[1,3],M[2,3]])
    return cls(R, t, CoordType.OpenGL)

  @classmethod
  def from_OpenGL(cls, R, t):
    return cls(R, t, CoordType.OpenGL)

  @classmethod
  def from_Unity3D_mat(cls, M):
    R = Rotation.from_matrix(M[:3,:3])
    t = np.matrix([M[0,3],M[1,3],M[2,3]])
    return cls(R, t, CoordType.Unity3D)

  @classmethod
  def from_Unity3D(cls, R, t):
    return cls(R, t, CoordType.Unity3D)

  @classmethod
  def from_FreeD_mat(cls, M):
    R = Rotation.from_matrix(M[:3,:3])
    t = np.matrix([M[0,3],M[1,3],M[2,3]])
    return cls(R, t, CoordType.FreeD)

  @classmethod
  def from_FreeD(cls, R, t):
    return cls(R, t, CoordType.FreeD)

  @classmethod
  def from_UE_mat(cls, M):
    R = Rotation.from_matrix(M[:3,:3])
    t = np.matrix([M[0,3],M[1,3],M[2,3]])
    return cls(R, t, CoordType.UE)

  @classmethod
  def from_UE(cls, R, t):
    return cls(R, t, CoordType.UE)

  @classmethod
  def from_INV_SLAM_mat(cls, M):
    R = Rotation.from_matrix(M[:3,:3])
    t = np.matrix([M[0,3],M[1,3],M[2,3]])
    return cls(R, t, CoordType.INV_SLAM)

  @classmethod
  def from_INV_SLAM(cls, R, t):
    return cls(R, t, CoordType.INV_SLAM)

  @staticmethod
  def as_mat(R, t):
    M = np.eye(4)
    M[:3,:3] = R.as_matrix()
    M[:3,3]  = np.array(t)
    return M

  def to_SLAM(self):
    return self.R, self.t

  def to_SLAM_mat(self):
    return self.as_mat(self.R, self.t)

  def to_OpenGL(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.OpenGL)

  def to_OpenGL_mat(self):
    R,t = self.convert(self.R, self.t, cTypeOut=CoordType.OpenGL)
    return self.as_mat(R,t)

  def to_Unity3D(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.Unity3D)

  def to_Unity3D_mat(self):
    R,t = self.convert(self.R, self.t, cTypeOut=CoordType.Unity3D)
    return self.as_mat(R, t)

  def to_FreeD(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.FreeD)

  def to_FreeD_mat(self):
    R,t = self.convert(self.R, self.t, cTypeOut=CoordType.FreeD)
    return self.as_mat(R,t)

  def to_UE(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.UE)

  def to_UE_mat(self):
    R,t = self.convert(self.R, self.t, cTypeOut=CoordType.UE)
    return self.as_mat(R,t)

  def to_INV_SLAM(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.INV_SLAM)

  def to_INV_SLAM_mat(self):
    R,t = self.convert(self.R, self.t, cTypeOut=CoordType.INV_SLAM)
    return self.as_mat(R,t)

  '''
    Computation
  '''
  # generate transform matrix of specific in and out types
  @staticmethod
  def computeM_in_out(cTypeIn, cTypeOut):
    if(cTypeIn==cTypeOut):
      return np.eye(3) # eye mat

    # Choose correct s^M_in 
    if(cTypeIn==CoordType.SLAM):
      s_M_in = np.eye(3)
    elif(cTypeIn==CoordType.OpenGL):
      s_M_in = s_M_opgl 
    elif(cTypeIn==CoordType.Unity3D):
      s_M_in = s_M_u3d 
    elif(cTypeIn==CoordType.FreeD):
      s_M_in = s_M_fd 
    elif(cTypeIn==CoordType.UE):
      s_M_in = s_M_ue 
    elif(cTypeIn==CoordType.INV_SLAM): # test
      s_M_in = s_M_invs 

    # Choose correct s^M_out 
    if(cTypeOut==CoordType.SLAM):
      s_M_out = np.eye(3)
    elif(cTypeOut==CoordType.OpenGL):
      s_M_out = s_M_opgl 
    elif(cTypeOut==CoordType.Unity3D):
      s_M_out = s_M_u3d 
    elif(cTypeOut==CoordType.FreeD):
      s_M_out = s_M_fd 
    elif(cTypeOut==CoordType.UE):
      s_M_out = s_M_ue 
    elif(cTypeOut==CoordType.INV_SLAM): # test
      s_M_out = s_M_invs 

    # out^M_in = (s^M_out)^-1(s^M_in)
    out_M_in = (s_M_out.transpose())*s_M_in

    return out_M_in

  # main conversion funcs, better wraped in the cls
  '''
  Maths Explication:

    T_out = out^M_in * T_in

    R_out = out^M_in * R_in * in^M_out    
  '''
  @classmethod
  def convert(cls, R, t, cTypeIn=CoordType.SLAM, cTypeOut=CoordType.SLAM):

    # Identical 
    if(cTypeIn==cTypeOut):
      return R, t

    R_m = R.as_matrix()
    t_v = t.transpose() # as col vector
    o_M_i = cls.computeM_in_out(cTypeIn, cTypeOut) 

    t_v_o = o_M_i * t_v
    R_m_o = o_M_i * R_m * (o_M_i.transpose())
    # R_m_o = o_M_i * R_m #* (o_M_i.transpose())

    R_o = Rotation.from_matrix(R_m_o) 
    t_o = t_v_o.transpose()

    return R_o, t_o
  
def unitTestConvertor():
  # init with 
  r = Rotation.from_quat([-0.5773503, -0.1924501, -0.1924501, 0.7698004])
  t = np.matrix([1,2,3])

  R_SLAM, t_SLAM = coordCvtor.from_SLAM(r,t).to_SLAM() 
  print ("R SLAM",R_SLAM.as_quat()) 
  print("\n")
  print ("t SLAM",t_SLAM)
  print("\n")

  assert((R_SLAM.as_matrix()==r.as_matrix()).all())
  assert((t_SLAM==t).all())

  # convert to U3D
  R_U3D, t_U3D = coordCvtor.from_SLAM(r,t).to_Unity3D() 
  print ("R U3D",R_U3D.as_quat())
  print("\n")
  print ("t U3D",t_U3D)
  print("\n")

  # convert to FreeD
  R_Fd, t_Fd = coordCvtor.from_Unity3D(R_U3D, t_U3D).to_FreeD() 
  print ("R FreeD",R_Fd.as_quat())
  print("\n")
  print ("t FreeD",t_Fd)
  print("\n")

  R_opgl, t_opgl = coordCvtor.from_FreeD(R_Fd, t_Fd).to_OpenGL() 
  print ("R OpenGL",R_opgl.as_quat())
  print("\n")
  print ("t OpenGL",t_opgl)
  print("\n")

  R_UE, t_UE = coordCvtor.from_OpenGL(R_opgl, t_opgl).to_UE() 
  print ("R UE",R_UE.as_quat())
  print("\n")
  print ("t UE",t_UE)
  print("\n")

  R_SLAM_, t_SLAM_ = coordCvtor.from_UE(R_UE, t_UE).to_SLAM() 
  print ("R SLAM back",R_SLAM_.as_quat())
  print("\n")
  print ("t SLAM back",t_SLAM_)
  print("\n")

  assert(np.linalg.norm(R_SLAM_.as_quat()-R_SLAM.as_quat())<10e-8)
  assert(np.linalg.norm(t_SLAM_-t_SLAM)<10e-8)

  print ("Unit test passed. \n")

# if __name__ == "__main__":
#   print("starting unit tests")
#   unitTestConvertor()


''' 

vizers and dumpers for poses in NeRF

'''
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from pytransform3d.trajectories import plot_trajectory
import pytransform3d.transformations as pytr
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def save_tpq(ltp, fname):
  with open(fname, 'w') as f:
    for idx, tp in enumerate(ltp):
      f.write(f"{idx} {tp[0]} {tp[1]} {tp[2]} {tp[3]} {tp[4]} {tp[5]} {tp[6]}\n")


def viz_poses(lposes, s=0.2, save_dir=None, save_dir_ext=None):
  ltp = []
  if (len(lposes)<2):
    return

  # assuming the original is opengl 
  lposes_SLAM = [coordCvtor.from_OpenGL_mat(pose).to_SLAM_mat() for pose in lposes]

  for p in lposes_SLAM:
    ort_R = Rotation.from_matrix(p[:3,:3]).as_matrix()
    Tp = pytr.transform_from(ort_R, p[:3,-1])
    Tpq = pytr.pq_from_transform(Tp, False)
    ltp.append(Tpq)

  print("saving ", len(ltp), " poses")
  if save_dir_ext!=None:
    save_tpq(ltp, save_dir_ext)

  ax = plot_trajectory(
    P=ltp, s=s, n_frames=len(lposes))

  # xlim, ylim, zlim = getlims(ltp,10)

  if save_dir==None:
    plt.show()
  else:
    plt.savefig(save_dir)

  