import numpy as np
import opticalflow3D

def randomize(vol, mean=0.0, std_dev=1.0):
  depth, height, width = vol.shape[:3]
  x_coords, y_coords, z_coords = np.meshgrid(range(width), range(height), range(depth))
  flattened_x_coords = x_coords.flatten()
  flattened_y_coords = y_coords.flatten()
  flattened_z_coords = z_coords.flatten()
  #print(np.max(flattened_z_coords), np.max(flattened_y_coords), np.max(flattened_x_coords))
  #print(flattened_x_coords.dtype)
  displacements_x = np.random.normal(mean, std_dev, flattened_x_coords.shape).astype(np.int32)
  displacements_y = np.random.normal(mean, std_dev, flattened_y_coords.shape).astype(np.int32)
  displacements_z = np.random.normal(mean, std_dev, flattened_z_coords.shape).astype(np.int32)
  #_d = 5
  #displacements_x = np.random.uniform(low=-_d, high=_d, size=flattened_x_coords.shape).astype(np.int32)
  #displacements_y = np.random.uniform(low=-_d, high=_d, size=flattened_y_coords.shape).astype(np.int32)
  #displacements_z = np.random.uniform(low=-_d, high=_d, size=flattened_z_coords.shape).astype(np.int32)
  print("min displacements", np.min(displacements_z), np.min(displacements_y), np.min(displacements_x))
  print("average abs(displacements)", np.average(np.abs(displacements_z)), np.average(np.abs(displacements_y)), np.average(np.abs(displacements_x)))
  print("max displacements", np.max(displacements_z), np.max(displacements_y), np.max(displacements_x))
  randomized_x_coords = flattened_x_coords + displacements_x
  randomized_y_coords = flattened_y_coords + displacements_y
  randomized_z_coords = flattened_z_coords + displacements_z
  #print("max displacements", np.max(randomized_z_coords), np.max(randomized_y_coords), np.max(randomized_x_coords))
  #randomized_x_coords = np.mod(randomized_x_coords, width)
  #randomized_y_coords = np.mod(randomized_y_coords, height)
  #randomized_z_coords = np.mod(randomized_z_coords, depth)
  randomized_x_coords = np.clip(randomized_x_coords, 0, width - 1) # Clip the randomized coordinates to stay within image bounds
  randomized_y_coords = np.clip(randomized_y_coords, 0, height - 1)
  randomized_z_coords = np.clip(randomized_z_coords, 0, depth - 1)
  #print(np.max(randomized_z_coords), np.max(randomized_y_coords), np.max(randomized_x_coords))
  #randomized_vol = np.ones_like(vol)*np.average(vol) #np.zeros_like(vol)
  randomized_vol = np.zeros_like(vol)
  #randomized_vol[...] = vol
  #randomized_vol[...] = 128
  randomized_vol[randomized_z_coords, randomized_y_coords, randomized_x_coords] = vol[flattened_z_coords, flattened_y_coords, flattened_x_coords]
  return randomized_vol

def _randomize(vol, max_distance=10):
    depth, height, width = image.shape[:3]
    #flow_x = np.random.normal(size=(height, width)) * max_distance
    #flow_y = np.random.normal(size=(height, width)) * max_distance
    flow_x = np.random.uniform(low=-1, high=1, size=(depth, height, width)) * max_distance
    flow_y = np.random.uniform(low=-1, high=1, size=(depth, height, width)) * max_distance
    flow_z = np.random.uniform(low=-1, high=1, size=(depth, height, width)) * max_distance
    #flow_x[...] = 0
    #flow_y[...] = 0
    #print(np.max(flow_x), np.min(flow_x), max_distance)
    flow = np.empty([height, width, 2], dtype=np.float32)
    flow[..., 0] = flow_y
    flow[..., 1] = flow_x
    print(np.max(flow), np.min(flow))
    randomized_image = motion_estimation.project(image, flow)
    return randomized_image.astype(np.uint8)

def shake(x, y, std_dev=1.0):
  displacements = np.random.normal(0, std_dev, len(x))
  #print(f"{np.min(displacements):.2f} {np.average(np.abs(displacements)):.2f} {np.max(displacements):.2f}", end=' ')
  return np.stack((y + displacements, x), axis=1)

def randomize(vol, mean=0.0, std_dev=1.0):
  print(vol.shape)
  print(std_dev)
  randomized_vol = np.empty_like(vol)
  
  # Randomization in X
  #values = np.arange(1, vol.shape[2]+1).astype(np.int32)
  values = np.arange(vol.shape[2]).astype(np.int32)
  for z in range(vol.shape[0]):
    print(z, end=' ', flush=True)
    for y in range(vol.shape[1]):
      #pairs = np.array(list(map(tuplify, values, range(len(values)))), dtype=np.int32)
      pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
      pairs = pairs[pairs[:, 0].argsort()]
      randomized_vol[z, y, values] = vol[z, y, pairs[:, 1]]
  vol = np.copy(randomized_vol)

  # Randomization in Y
  values = np.arange(vol.shape[1]).astype(np.int32)
  for z in range(vol.shape[0]):
    print(z, end=' ', flush=True)
    for x in range(vol.shape[2]):
      #pairs = np.array(list(map(tuplify, values, range(len(values)))), dtype=np.int32)
      pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
      pairs = pairs[pairs[:, 0].argsort()]
      randomized_vol[z, values, x] = vol[z, pairs[:, 1], x]
  vol = np.copy(randomized_vol)

  # Randomization in Z
  values = np.arange(vol.shape[0]).astype(np.int32)
  for y in range(vol.shape[1]):
    print(y, end=' ', flush=True)
    for x in range(vol.shape[2]):
      #pairs = np.array(list(map(tuplify, values, range(len(values)))), dtype=np.int32)
      pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
      pairs = pairs[pairs[:, 0].argsort()]
      randomized_vol[values, y, x] = vol[pairs[:, 1], y , x]

  return randomized_vol

def project_A_to_B(farneback, block_size, A, B):
  output_vz, output_vy, output_vx, output_confidence = farneback.calculate_flow(A, B,
                                                                              start_point=(0, 0, 0),
                                                                              total_vol=(A.shape[0], A.shape[1], A.shape[2]),
                                                                              sub_volume=block_size,
                                                                              overlap=(8, 8, 8),
                                                                              threadsperblock=(8, 8, 8)
                                                                             )
  print("min flow", np.min(output_vx), np.min(output_vy), np.min(output_vz))
  print("average abs(flow)", np.average(np.abs(output_vx)), np.average(np.abs(output_vy)), np.average(np.abs(output_vz)))
  print("max flow", np.max(output_vx), np.max(output_vy), np.max(output_vz))
  #output_vx[...] = 0
  #output_vy[...] = 0
  #output_vz[...] = 0
  projection = opticalflow3D.helpers.generate_inverse_image(A, output_vx, output_vy, output_vz)
  return projection

def filter(farneback, block_size, noisy_vol, N_iters=50, RS_sigma=2.0, RS_mean=0.0):
  acc_vol = np.zeros_like(noisy_vol, dtype=np.float32)
  acc_vol[...] = noisy_vol
  for i in range(N_iters):
    print(f"iter={i}")
    denoised_vol = acc_vol/(i+1)
    #randomized_noisy_vol = randomize(noisy_vol, max_distance=5)
    randomized_noisy_vol = randomize(noisy_vol, mean=0, std_dev=RS_sigma)
    #print("sum(randomized_noisy-noisy)", np.sum((randomized_noisy_vol-noisy_vol)*(randomized_noisy_vol-noisy_vol)))
    #print("sum(denoised-randomized_noisy)", np.sum((denoised_vol-randomized_noisy_vol)*(denoised_vol-randomized_noisy_vol)))
    randomized_and_compensated_noisy_vol = project_A_to_B(farneback, block_size, A=denoised_vol, B=randomized_noisy_vol)
    #plt.imshow(randomized_and_compensated_noisy_vol[15], cmap="gray")
    #plt.show()
    #randomized_and_compensated_noisy_vol = np.zeros_like(randomized_noisy_vol)
    #randomized_and_compensated_noisy_vol[...] = randomized_noisy_vol
    #print("sum(noisy)", np.sum(noisy_vol))
    #print("sum(denoised)", np.sum(denoised_vol))
    #print("sum(randomized_and_compensated_noisy)", np.sum(randomized_and_compensated_noisy_vol))
    #print("sum(randomized_noisy)", np.sum(randomized_noisy_vol))
    #print("sum(acc)", np.sum(acc_vol))
    #print("sum(randomized_and_compensated_noisy-randomized_noisy)", np.sum((randomized_and_compensated_noisy_vol-randomized_noisy_vol)*(randomized_and_compensated_noisy_vol-randomized_noisy_vol)))
    #print("sum(randomized_and_compensated_noisy-noisy)", np.sum((randomized_and_compensated_noisy_vol-noisy_vol)*(randomized_and_compensated_noisy_vol-noisy_vol)))
    #print(np.sum((randomized_and_compensated_noisy_vol-randomized_noisy_vol)*(randomized_and_compensated_noisy_vol-randomized_noisy_vol)))
    acc_vol += randomized_and_compensated_noisy_vol
  denoised_vol = acc_vol/(N_iters + 1)
  return denoised_vol