import numpy as np


def per_channel_color_histogram(interval, r, g, b):

    hist_r=np.bincount(r,minlength=256)
    hist_g=np.bincount(g,minlength=256)
    hist_b=np.bincount(b,minlength=256)

    hist_r = hist_r.reshape(-1, interval)
    hist_g = hist_g.reshape(-1, interval)
    hist_b = hist_b.reshape(-1, interval)

    hist_r = np.sum(hist_r, axis=1)
    hist_g = np.sum(hist_g, axis=1)
    hist_b = np.sum(hist_b, axis=1)

    return hist_r, hist_g, hist_b

def threed_color_histogram(region_reshaped, num_of_bins_per_channel, interval):

    region_reshaped=region_reshaped//interval
    r,g,b=region_reshaped[:,2],region_reshaped[:,1],region_reshaped[:,0]
    hist_threed_array = np.bincount(r * num_of_bins_per_channel ** 2 + g * num_of_bins_per_channel + b, minlength=num_of_bins_per_channel**3)
    hist_threed = hist_threed_array.reshape((num_of_bins_per_channel,num_of_bins_per_channel,num_of_bins_per_channel))
    return hist_threed.ravel()

def rgb_to_hsv_color_space_calculation(rh,gs,bv):

    epsilon=np.finfo(float).eps
    a=np.zeros((rh.shape[0],3))
    a[:,0],a[:,1],a[:,2]=rh,gs,bv
    max_index=np.argmax(a, axis=1)
    min_index=np.argmin(a, axis=1)
    row_indices = np.arange(len(max_index))
    c_max=a[row_indices, max_index]
    c_min=a[row_indices, min_index]
    delta_c=c_max-c_min
    v=c_max
    s = np.where(v > 0, delta_c / (c_max+epsilon), 0)
    h1 = np.where(max_index == 0, (((a[:, 0] - a[:, 1]) / (delta_c+epsilon)) % 6)/6, 0)
    h2 = np.where(max_index == 1, (((a[:, 1] - a[:, 2]) / (delta_c+epsilon))  +2)/6 ,0)
    h3 = np.where(max_index == 2, (((a[:, 2] - a[:, 0]) / (delta_c+epsilon))  +4)/6, 0)
    h=h1+h2+h3
    h= np.where(delta_c==0, 0,h)
    h,s,v=h*255,s*255,v*255
    return h.astype(int),s.astype(int),v.astype(int)

def grid_based_feature_extraction(image, n_by_n, color_histogram_type, color_space_type, quantization_interval):

    num_of_bins = 256 // quantization_interval
    M = image.shape[0] // n_by_n
    N = image.shape[1] // n_by_n
    regions = [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]
    histograms = []
    for region in regions:
        region_height = region.shape[0]
        region_reshaped = region.reshape(region_height ** 2, 3)
        if color_histogram_type=="per-channel":
            rh, gs, bv = region_reshaped[:, 2], region_reshaped[:, 1], region_reshaped[:, 0]
            if color_space_type == "hsv":
                rh,gs,bv=rgb_to_hsv_color_space_calculation(rh/255,gs/255,bv/255)
            hist_rh,hist_gs,hist_bv= per_channel_color_histogram(quantization_interval, rh, gs, bv)
            hist_rh= hist_rh / np.sum(hist_rh)
            hist_gs = hist_gs / np.sum(hist_gs)
            hist_bv= hist_bv / np.sum(hist_bv)
            histograms.append(hist_rh)
            histograms.append(hist_gs)
            histograms.append(hist_bv)
        elif color_histogram_type == "3d":
            if color_space_type == "hsv":
                rh, gs, bv = region_reshaped[:, 2], region_reshaped[:, 1], region_reshaped[:, 0]
                region_reshaped[:, 2], region_reshaped[:, 1], region_reshaped[:, 0] = rgb_to_hsv_color_space_calculation(rh/255, gs/255, bv/255)
            hist_threed = threed_color_histogram(region_reshaped, num_of_bins, quantization_interval)
            hist_threed = hist_threed / np.sum(hist_threed)
            histograms.append(hist_threed)
    return histograms
