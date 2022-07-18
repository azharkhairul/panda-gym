import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import cv2
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
# def distance(a, b):   #original distance function
#     assert a.shape == b.shape

#     return np.linalg.norm(b-a, axis=-1)
#     # return np.array(result)
def distance(a, b):
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)
# def distance(a, b):      #sparse 
#     assert a.shape == b.shape
#     result = a*b
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

        # self.width=45
        # self.height=30
        # self.width=23
        # self.height=15
        self.width=13
        self.height=8
        view_matrix1 = self.physics_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0 , 0.0),
            distance=0.3,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self.physics_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.width) / self.height, nearVal=0.1, farVal=100.0
        )
        
        (_, _,px1, depth1, mask1) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix,
            # renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        view_matrix2 = self.physics_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0 , 0.0),
            distance=0.3,
            yaw=135,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )

        (_, _, px2, depth2, mask2) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix2,
            projectionMatrix=proj_matrix,
            # renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        # view_matrix3 = self.physics_client.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=(0.0, 0.0 , 0.0),
        #     distance=0.3,
        #     yaw=-45,
        #     pitch=-30,
        #     roll=0,
        #     upAxisIndex=2,
        # )
        
        # (_, _, px3, depth3, mask3) = p.getCameraImage(
        #     width=self.width,
        #     height=self.height,
        #     viewMatrix=view_matrix3,
        #     projectionMatrix=proj_matrix,
        #     # renderer=p.ER_BULLET_HARDWARE_OPENGL,
        # )data,
        return px1,px2,mask1,mask2



def thresh_callback(img):
    # width=45
    # height=30
    width=17
    height=11

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
    # px1,px2,px3,mask1,mask2,mask3 = get_view(self)
    px1,px2,mask1,mask2 = get_view(self)
    
    # rgb1 = Image.fromarray(px1) 
    # rgb1.save('CAMMMEERRAAA11111.png')
    # rgb2 = Image.fromarray(px2) 
    # rgb2.save('../dkdc/CAMMMEERRAAA22222.png')

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
    # BW_obj1.save("goal121.png")

    mask_obj2 = Image.fromarray(mask2) 
    new_image2 = []
    for item in mask_obj2.getdata():

        if (item == mode ):#object        
            new_image2.append((1))
        else:
            new_image2.append((0))

    
    # mask_obj3 = Image.fromarray(mask3) 
    # new_image3 = []
    # for item in mask_obj3.getdata():

    #     if (item == mode ):#object        
    #         new_image3.append((1))
    #     else:
    #         new_image3.append((0))

    return new_image1, new_image2 #, new_image3

def eef_binary(self, a):
    # self.width=34
    # self.height=23
    self.width=17
    self.height=11

    ee_pos = np.array(self.sim.get_link_position("panda", 11))
    xA, yA, zA = ee_pos
    zA = zA +0.1 # make the camera a little higher than the robot

    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[xA+0.15, yA, zA-0.06],
                        cameraTargetPosition=[xA, yA, zA -0.06],
                        cameraUpVector=[0, 0, 1]
                    )
    proj_matrix = self.physics_client.computeProjectionMatrixFOV(
        fov=60, aspect=float(self.width) / self.height, nearVal=0.1, farVal=100.0
    )
    
    (_, _, px, depth, mask) = p.getCameraImage(
        width=self.width,
        height=self.height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        # renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    if a == "obj":
        mode = 3.0 #object
    else:
        mode = 4.0 #goal

    mask = Image.fromarray(mask) 
    new_image = []
    for item in mask.getdata():
        if (
            item == mode #object              
        ):
            new_image.append((1))
        else:
            new_image.append((0))
    # BW_obj1 = Image.new('1', (self.width, self.height))
    # BW_obj1.putdata(new_image)
    # BW_obj1.save("eeff.png")
    return new_image

def compute_M(data):
    data = np.array(data)
    cols = np.arange(data.size)
    return csr_matrix((cols, (np.ravel(data), cols)),shape=(data.max() + 1, data.size))

def get_indices_sparse(self,a, num):
    if a == "obj":
        new_image1, new_image2 = mask2binary(self, a)
    else:
        new_image1, new_image2 = mask2binary(self, 'goal')

    if num == 1:
        data = new_image1
    else:
        data = new_image2

    data = np.array(data)
    M = compute_M(data)
    return [np.mean(np.unravel_index(row.data, data.shape),1) for R,row in enumerate(M) if R>0]