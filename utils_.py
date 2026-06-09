import numpy as np
import hashlib
from noise import pnoise2
from PIL import Image


####################################################################################################
### data marking 2):  injecting procedural noise (adapted from Co et al. in CCS'19, https://github.com/kenny-co/procedural-advml )
####################################################################################################
def perlin(size, period, octave, freq_sine, lacunarity = 2, base =0):
    def normalize(vec):
        vmax = np.amax(vec)
        vmin  = np.amin(vec)
        return (vec - vmin) / (vmax - vmin)
    noise = np.empty((size[1], size[2]), dtype = np.float32)
    for x in range(size[1]):
        for y in range(size[2]):
            noise[x][y] = pnoise2(x / period, y / period, octaves = octave, lacunarity = lacunarity, base=base)  
    # Sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    return normalize(noise)


def perturb(img, noise, norm):
    '''
    Perturb image and clip to maximum perturbation norm

    img              image with pixel range [0, 1]
    noise           noise with pixel range [-1, 1]
    norm           L-infinity norm constraint
    '''
    noise = np.sign((noise - 0.5) * 2) * norm
    noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
    return (img + noise)


def colorize(img, color = [1, 1, 1]):
    ### Visualize Image ###
    '''
    Color image

    img              has dimension 2 or 3, pixel range [0, 1]
    color            is [a, b, c] where a, b, c are from {-1, 0, 1}
    '''
    if img.ndim == 2: # expand to include color channels
        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, axis=2)  # replicate grayscale to RGB: (H,W,1) -> (H,W,3)
    color = np.array(color).reshape(1, 1, 3)  # reshape for proper broadcasting
    return (img - 0.5) * color + 0.5 # output pixel range [0, 1]


def get_unique_random_seed(seeded_sample):
    # generate a unique random seed from a seeded_sample
    data = seeded_sample.flatten()
    hashobj = hashlib.md5
    hashes = hashobj(data)
    seed = np.frombuffer(hashes.digest(), dtype='uint32')
    return seed


def add_noise_to_sample(pil_image):
    # pil_image: PIL Image on which we will add perlin noise
    
    # Convert PIL Image to numpy array (HxWx3 -> CxHxW)
    pil_image = pil_image.convert('RGB')
    sample_to_be_marked = np.asarray(pil_image, dtype=np.uint8) / 255.0
    sample_to_be_marked = sample_to_be_marked.transpose((2, 0, 1)).astype(np.float32)

    seed = get_unique_random_seed(sample_to_be_marked) # generate deterministic random perlin noise for each sample
    rstate = np.random.RandomState(seed) 
    norm = 8 / 255.

    period= rstate.randint(low=30, high= 60,)   # generate random perlin noise
    octave = rstate.randint(low=1, high= 5)     
    freq_sine = rstate.randint(low=20, high= 60)         
    noise = perlin(size = sample_to_be_marked.shape, period = period, octave = octave, freq_sine = freq_sine) 
    noise =  colorize(noise) 
    sample_to_be_marked = perturb(img = sample_to_be_marked.transpose((1,2,0)), norm = norm, noise=noise )
    sample_to_be_marked = sample_to_be_marked.transpose((2, 0, 1))
    
    # Convert back to PIL Image (CHW -> HWC)
    sample_to_be_marked = np.clip(sample_to_be_marked * 255, 0, 255).astype(np.uint8)
    sample_to_be_marked = sample_to_be_marked.transpose((1, 2, 0))  # CHW -> HWC
    return Image.fromarray(sample_to_be_marked)


def gen_unique_outlier_pattern(sample_as_random_seed, ood_pattern='color_stripes', outlierDataSet=None):
    # generate a unique outlier pattern for image blending
    
    # sample_as_random_seed:  we use an image to generate a unique random seed, which will be used to create a unique outlier pattern
    #                         for each target user's data, we apply the *same* outlier pattern for data marking, 
    #                               hence we need a random seed to create the same pattern
    # ood_pattern: types of outlier pattern (e.g., random color stripes, or data from an OOD dataset)
    # outlierDataSet: this is used only if the ood_pattern is ``tinyimagenet'' or ``celeba'', in which case it will contain data from either dataset


    new_sample = sample_as_random_seed.transpose((1, 2, 0))
    size = new_sample.shape 
    seed = get_unique_random_seed(new_sample) # generate a unique random seed from the seeded sample
    rstate = np.random.RandomState(seed) 
    x = []
    outlier = []
    if(ood_pattern == 'color_stripes'):
        # a list of common color to create the outlier feature with color stripes
        color_list = [
            [255, 255, 255], # white
            [255, 0, 0], # red
            [0, 255, 0], # blue
            [255, 255, 0], # yellow
            [0, 255, 0], # green 
            [255, 0, 255], # purple 
            [255, 165, 0], # orange
            [0, 0, 0] ,  # black
            [128, 128, 128], # gray
            [165, 42, 42], #brown
            [255,192,203], # pink
        ]
        num_color_stripes=16
        color_list = np.array(color_list)
        color_list = color_list / 8. # This is for reducing the color difference among the different color stripes
        color_list = color_list + 128 # Adjust the brightness

        color_stripes_size = int(size[1] / num_color_stripes)
        random_colour = rstate.randint(low=0, high= len(color_list), size =num_color_stripes) 
        outlier = np.zeros( shape=size )
        for i in range(num_color_stripes):
            outlier[ i*color_stripes_size : (i+1)*color_stripes_size, :, : ] = color_list[ random_colour[i] ]
        outlier = outlier.transpose((2, 0, 1))
        outlier /= 255.
        x.append(outlier)
    elif(ood_pattern == 'tinyimagenet' or ood_pattern == 'celeba'):
        # use samples from an ood dataset as the outlier features
        idx = rstate.randint(low=0, high= len(outlierDataSet)) 
        outlier = outlierDataSet[idx][0]
        x.append(outlier.numpy())
    return x