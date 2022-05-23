import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import cv2
import numpy as np
from PIL import Image

# def distance(a, b):
#     # print(a.shape)
#     # print(b.shape)
#     assert a.shape == b.shape
#     result = a-b
#     return np.linalg.norm(result, axis=-1)

def threshold_convergence(input):
    k = 5  #the enlargement scale
    # a = 0.2 #speed of convergence for step-down
    b = 2.5   #speed of convergence for exponential
    c = input
    n = 50.0 #number of success
    e = 0.04 #original size of the target
    div = c/n

    size = (1+(k-1)/(b**(div)))*(e)  #exponential convergence
    # size = (1+(k-1)(1-(c/n)))(e)    #linearly convergence
    # size = (1+(k-1)(1-a(c/n)))(e)   #step-down convergence
    
    return size

def comparee(a1,b1,a2,b2):
    penalty1 = 0
    penalty2 = 0
    if (sum(abs(a1-a2))) > 0.01:
        penalty1=20
    if (sum(abs(b1-b2))) > 0.01:
        penalty2=20

    return -(penalty1+penalty2)

def compareemoreobj(a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2):
    penalty1 = 0
    penalty2 = 0
    penalty3 = 0
    penalty4 = 0
    penalty5 = 0
    penalty6 = 0
    if (sum(abs(a1-a2))) > 0.01:
        penalty1=20
    if (sum(abs(b1-b2))) > 0.01:
        penalty2=20    
    if (sum(abs(c1-c2))) > 0.01:
        penalty3=20
    if (sum(abs(d1-d2))) > 0.01:
        penalty4=20   
    if (sum(abs(e1-e2))) > 0.01:
        penalty5=20
    if (sum(abs(f1-f2))) > 0.01:
        penalty6=20
    return -(penalty1+penalty2+penalty3+penalty4+penalty5+penalty6)

def get_view(self):
        # self.width=90
        # self.height=60
        # self.width=68
        # self.height=45
        self.width=45
        self.height=30
        # self.width=23
        # self.height=15
        view_matrix1 = self.physics_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0 , 0.15),
            distance=0.6,
            yaw=90,
            pitch=-50,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix1 = self.physics_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.width) / self.height, nearVal=0.1, farVal=100.0
        )
        
        (_, _,px1, depth1, mask1) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix1,
            # renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        view_matrix2 = self.physics_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0 , 0.15),
            distance=0.7,
            yaw=0,
            pitch=-0,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix2 = self.physics_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.width) / self.height, nearVal=0.1, farVal=100.0
        )
        
        (_, _, px2, depth2, mask2) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix2,
            projectionMatrix=proj_matrix2,
            # renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return px1,px2,mask1,mask2


# def distance(a, b):
#     # print(a.shape)
#     # print(b.shape)
    
#     if np.any(a != None) and np.any(b!=None):
#         assert a.shape == b.shape
#         result = a-b
#         return np.linalg.norm(result, axis=-1)
#     else:
#         return 0
def distance(a, b):
    # print(a.shape)
    # print(b.shape)
    assert a.shape == b.shape
    result = a*b
    return np.linalg.norm(result, axis=-1)
    # return (sum(result))

def thresh_callback(img):
    width=45
    height=30
    threshold = 100
    shi = np.array(img,dtype='uint8')
    img = np.reshape(shi, (height, width))
        # print(cam2)
    # canny_output = cv2.Canny(img, 0.5, 1.5)
    # print(np.shape(canny_output))
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    # print(centers[0][0],',', centers[0][1])
    # print('contor',len(contours))
    # print('sdadfasfdafasfasf = ', len(contours))
    if len(contours) !=0:
        for i in range(len(contours)):
            # print(i,centers[i][0],',', centers[i][1])
            
            return np.array([centers[i][0],centers[i][1]])
                    # if centers[i][0] == 0.0 and centers[i][1] ==0.0:
            #     return np.array([0.0,0.0])

    else:
        return None


def mask2binary(self, a):
    px1,px2,mask1,mask2 = get_view(self)

    if a == "obj":
        mode = 3.0 #object
    else:
        mode = 4.0 #goal

    mask_obj1 = Image.fromarray(mask1) 
    new_image1 = []
    for item in mask_obj1.getdata():
        if (
            item == mode #object              
        ):
            new_image1.append((1))
        else:
            new_image1.append((0))

    # BW_obj1 = Image.new('1', (self.width, self.height))
    # BW_obj1.putdata(new_image1)
    # BW_obj1.save("goal1.jpg")

    mask_obj2 = Image.fromarray(mask2) 
    new_image2 = []
    for item in mask_obj2.getdata():

        if (item == mode ):#object        
            new_image2.append((1))
        else:
            new_image2.append((0))

    return new_image1, new_image2