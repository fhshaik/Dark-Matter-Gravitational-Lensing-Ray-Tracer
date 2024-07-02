import taichi as ti
from PIL import Image
import numpy as np


ti.init()

pil_image = Image.open("skyMap.jpg")
width, height = pil_image.size
image_np = np.array(pil_image)
image_taichi = ti.field(dtype=ti.u8, shape=(height, width, 3))
image_taichi.from_numpy(image_np)


horizontal_resolution = 1280
vertical_resolution = 720
pixels = ti.field(ti.u8, shape=(horizontal_resolution, vertical_resolution, 3))


@ti.func
def generate_random_orthonormal_triad():
    # Generate a random vector uniformly distributed on the unit sphere
    theta = np.random.uniform(0, np.pi)
    phi = np.arccos(np.random.uniform(0, 1) - 1)
    
    normal = ti.Vector([
        ti.sin(phi) * ti.cos(theta),
        ti.sin(phi) * ti.sin(theta),
        ti.cos(phi)
    ])

    # Generate a random vector in the plane orthogonal to 'normal'
    theta = np.random.uniform(0, 2 * np.pi)
    tangent1 = ti.Vector([
        ti.cos(theta),
        ti.sin(theta),
        0.0
    ])
    tangent1 -= normal * normal.dot(tangent1)
    tangent1 = tangent1.normalized()

    # Cross product to find the third orthogonal vector
    tangent2 = normal.cross(tangent1).normalized()

    return normal, tangent1, tangent2

@ti.func
def raytrace(vector, darkmatter_vector):

    vector = vector.normalized()
    darkmatter_vector = darkmatter_vector.normalized()

    #calculation of distortion
    angle_rad = ti.acos(ti.math.dot(vector, darkmatter_vector))
    C = 100
    k = 1
    d = 5000
    b = ti.abs(d*ti.tan(angle_rad))
    distortion_factor = C * (2 * k**3 - (2 * k**3 + 2 * b * k**2 + b**2 * k) * ti.exp(-b / k)) / b
    rotation_axis = ti.math.cross(vector,darkmatter_vector)
    # print(rotation_axis)
    axis = rotation_axis.normalized()
    # print(axis)

    #print("hellloooo dosto", axis,rotation_axis,vector)
    #print(angle_rad)
    #print(distortion_factor*2*np.pi)
    #print(vector)
    cos_theta = ti.cos(distortion_factor*2*np.pi)
    sin_theta = ti.sin(distortion_factor*2*np.pi)

    vector = vector * cos_theta + ti.math.cross(axis,vector) * sin_theta + axis*ti.math.dot(vector,axis) * (1 - cos_theta)


    #equirectangular projection
    vector = vector.normalized()
    # Calculate latitude (phi) and longitude (lambda)
    zVal = (vector[2]+1)/2  # latitude
    xyVector = vector - ti.Vector([0,0,vector[2]])
    xyVector.normalized()
    angle = ti.atan2(xyVector[1], xyVector[0])+np.pi#ti.acos(ti.math.dot(xyVector, ti.Vector([1,0,0])))
    #print(image_taichi.shape)
    #print(phi, lambda_)

    x = ((angle/(2*np.pi)))*image_taichi.shape[1]
    y = (zVal)*image_taichi.shape[0]
    # print(image_taichi.shape)
    # print(x,y)

    x = ti.min(ti.max(x, 0), image_taichi.shape[1] - 1)
    y = ti.min(ti.max(y, 0), image_taichi.shape[0] - 1)

    #print(x,y)
    # Convert coordinates to integers
    shape1 = ti.cast(x, ti.i32)
    shape2 = ti.cast(y, ti.i32)

    return ti.Vector([image_taichi[shape2, shape1, 0], image_taichi[shape2, shape1, 1], image_taichi[shape2, shape1, 2]])

@ti.kernel
def camera():
    sample_size = 5
    focal_length = 1
    viewport_height = 2.0
    
    ratio = pixels.shape[0]/pixels.shape[1]
    viewport_width = viewport_height*ratio

    pixel_delta_u = viewport_width / pixels.shape[0]
    pixel_delta_v = viewport_height / pixels.shape[1]
    ihat, jhat, khat = generate_random_orthonormal_triad()
    #ihat, jhat, khat = ti.Vector([0,-1,0]), ti.Vector([0,0,1]), ti.Vector([1,0,0])
    upperleft = ihat*focal_length - viewport_width*jhat*0.5 + viewport_height*khat*0.5
    pixel_00 = upperleft + 0.5*pixel_delta_u*jhat - 0.5*pixel_delta_v*khat
    darkmatter_vector = pixel_00 + (pixels.shape[0]*ti.random()*jhat*pixel_delta_u) - (pixels.shape[1]*ti.random()*khat*pixel_delta_v)
    for x, y in ti.ndrange((0, pixels.shape[0]), (0, pixels.shape[1])):
        vector = pixel_00 + x*pixel_delta_u*jhat - y*pixel_delta_v*khat
        color_vector = ti.Vector([0.0, 0.0, 0.0])
        for k in range(sample_size):
            new_vector = vector + (ti.random()-0.5)*pixel_delta_u*jhat + (ti.random()-0.5)*pixel_delta_v*khat
            color_vector = color_vector+raytrace(new_vector,darkmatter_vector)
        color_vector = color_vector/sample_size
        pixels[x,y,0], pixels[x,y,1], pixels[x,y,2] = int(color_vector[0]), int(color_vector[1]), int(color_vector[2])
    return

camera()
gui = ti.GUI("Image", (pixels.shape[0],pixels.shape[1]))

while gui.running:
    gui.set_image(pixels)
    gui.show("darkmatter.jpg")